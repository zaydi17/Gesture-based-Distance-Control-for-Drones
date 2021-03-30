import time
from collections import deque

import cv2
import math
import queue
import threading
import traceback
import curses
import numpy as np
from joblib import load
import logging
from detect import Detector

import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, moveBy, CancelMoveBy
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
from olympe.messages.ardrone3.PilotingSettings import MaxTilt
from olympe.messages.ardrone3.SpeedSettings import MaxVerticalSpeed, MaxRotationSpeed

olympe.log.update_config({"loggers": {"olympe": {"level": "ERROR"}}})
logging.basicConfig(filename='log.log', level=logging.INFO, format="%(asctime)s.%(msecs)03d;%(levelname)s;%(message)s;",
                    datefmt='%Y-%m-%d,%H:%M:%S')
logging.info("message;distance;time(seconds)")

DRONE_IP = "192.168.42.1"
event_time = time.time()


# get euclidean distance between two points in instances
def get_point_distance(instances, one, two):
    p1 = instances[0][one]
    p2 = instances[0][two]
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# get a point as tuple
def get_point(instances, number):
    point = instances[0][number]
    return int(point[0]), int(point[1])


# remove outliers from two-dimensional array a
def remove_outliers(a):
    if len(a.shape) == 2:
        q1 = np.percentile(a, 25, interpolation='midpoint', axis=0)
        q3 = np.percentile(a, 75, interpolation='midpoint', axis=0)
        iqr = q3 - q1
        out1 = np.where(a + (1.5 * iqr) - q1 < 0)[0]
        out2 = np.where(a - (1.5 * iqr) - q3 > 0)[0]
        out = np.unique(np.append(out1, out2))
        a = np.delete(a, out, axis=0)
    return a


class FlightListener(olympe.EventListener):

    @olympe.listen_event(FlyingStateChanged())
    def onStateChanged(self, event, scheduler):
        # log flight state
        logging.error("{};{};{:.3f}".format(event.message.name, event.args["state"], time.time() - event_time))


def get_distance_height(height):
    return 211.08 * math.pow(height, -1.045)


def get_distance_width(width):
    return 223.05 * math.pow(width, -1.115)


class MainClass(threading.Thread):

    def __init__(self):
        # Create the olympe.Drone object from its IP address
        self.drone = olympe.Drone(DRONE_IP)
        # subscribe to flight listener
        listener = FlightListener(self.drone)
        listener.subscribe()
        self.last_frame = np.zeros((1, 1, 3), np.uint8)
        self.frame_queue = queue.Queue()
        self.flush_queue_lock = threading.Lock()
        self.detector = Detector()
        self.keypoints_image = np.zeros((1, 1, 3), np.uint8)
        self.keypoints = deque(maxlen=5)
        self.faces = deque(maxlen=10)
        self.f = open("distances.csv", "w")
        self.face_distances = deque(maxlen=10)

        self.image_width = 1280
        self.image_height = 720
        self.half_face_detection_size = 150

        self.poses_model = load("models/posesmodel.joblib")
        self.pose_predictions = deque(maxlen=5)

        self.pause_finding_condition = threading.Condition(threading.Lock())
        self.pause_finding_condition.acquire()
        self.pause_finding = True
        self.person_thread = threading.Thread(target=self.fly_to_person)
        self.person_thread.start()

        # flight parameters in meter
        self.flight_height = 0.0
        self.max_height = 1.0
        self.min_dist = 1.5

        # keypoint map
        self.nose = 0
        self.left_eye = 1
        self.right_eye = 2
        self.left_ear = 3
        self.right_ear = 4
        self.left_shoulder = 5
        self.right_shoulder = 6
        self.left_elbow = 7
        self.right_elbow = 8
        self.left_wrist = 9
        self.right_wrist = 10
        self.left_hip = 11
        self.right_hip = 12
        self.left_knee = 13
        self.right_knee = 14
        self.left_ankle = 15
        self.right_ankle = 16

        # person distance
        self.eye_dist = 0.0

        # save images
        self.save_image = False
        self.image_counter = 243
        self.pose_file = open("poses.csv", "w")
        super().__init__()
        super().start()

    def start(self):
        self.drone.connect()

        # Setup your callback functions to do some live video processing
        self.drone.set_streaming_callbacks(
            raw_cb=self.yuv_frame_cb,
            start_cb=self.start_cb,
            end_cb=self.end_cb,
            flush_raw_cb=self.flush_cb,
        )
        # Start video streaming
        self.drone.start_video_streaming()
        # set maximum speeds
        print("rotation", self.drone(MaxRotationSpeed(1)).wait().success())
        print("vertical", self.drone(MaxVerticalSpeed(0.1)).wait().success())
        print("tilt", self.drone(MaxTilt(5)).wait().success())

    def stop(self):
        # Properly stop the video stream and disconnect
        self.drone.stop_video_streaming()
        self.drone.disconnect()

    def yuv_frame_cb(self, yuv_frame):
        """
        This function will be called by Olympe for each decoded YUV frame.

            :type yuv_frame: olympe.VideoFrame
        """
        yuv_frame.ref()
        self.frame_queue.put_nowait(yuv_frame)

    def flush_cb(self):
        with self.flush_queue_lock:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait().unref()
        return True

    def start_cb(self):
        pass

    def end_cb(self):
        pass

    def show_yuv_frame(self, window_name, yuv_frame):
        # the VideoFrame.info() dictionary contains some useful information
        # such as the video resolution
        info = yuv_frame.info()
        height, width = info["yuv"]["height"], info["yuv"]["width"]

        # yuv_frame.vmeta() returns a dictionary that contains additional
        # metadata from the drone (GPS coordinates, battery percentage, ...)

        # convert pdraw YUV flag to OpenCV YUV flag
        cv2_cvt_color_flag = {
            olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
        }[info["yuv"]["format"]]

        # yuv_frame.as_ndarray() is a 2D numpy array with the proper "shape"
        # i.e (3 * height / 2, width) because it's a YUV I420 or NV12 frame

        # Use OpenCV to convert the yuv frame to RGB
        cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)
        # show video stream
        cv2.imshow(window_name, cv2frame)
        cv2.moveWindow(window_name, 0, 500)

        # show other windows
        self.show_face_detection(cv2frame)
        self.show_keypoints()
        cv2.waitKey(1)

    def show_keypoints(self):
        if len(self.keypoints) > 2:
            # display eye distance
            cv2.putText(self.keypoints_image, 'Distance(eyes): ' + "{:.2f}".format(self.eye_dist) + "m", (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)
            # display nose height
            cv2.putText(self.keypoints_image, 'Nose: ' + "{:.2f}".format(
                get_point(np.average(self.keypoints, axis=0), self.nose)[1] / self.image_height)
                        , (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow("keypoints", self.keypoints_image)
        cv2.moveWindow("keypoints", 500, 0)

    def show_face_detection(self, cv2frame):
        # get sub image
        img = self.get_face_detection_crop(cv2frame)
        # get face rectangle
        face, img = self.detector.detect_face(img)
        if face.size > 0:
            self.faces.append(face)
            width = face[2] - face[0]
            height = face[3] - face[1]
            # get distances for rectangle width and height
            width = get_distance_width(width)
            height = get_distance_height(height)
            # display distances
            cv2.putText(img, 'width: ' + "{:.2f}".format(width), (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)
            cv2.putText(img, 'height: ' + "{:.2f}".format(height), (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)
            cv2.putText(img, 'mean: ' + "{:.2f}".format(np.mean(np.array([width, height]))), (0, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)
            # append outlier free distance to log
            self.face_distances.append(self.get_face_distance())
        elif len(self.faces) > 0:
            # remove from dequeue if no face was detected
            self.faces.popleft()
        # show detection
        cv2.imshow("face", img)
        cv2.moveWindow("face", 0, 0)

    # get 300*300 crop of frame based on nose location or from the middle
    def get_face_detection_crop(self, cv2frame):
        if len(self.keypoints) > 0:
            x, y = get_point(np.array(self.keypoints, dtype=object)[-1], self.nose)
        else:
            x = cv2frame.shape[1] / 2
            y = cv2frame.shape[0] / 2
        x = max(self.half_face_detection_size, x)
        y = max(self.half_face_detection_size, y)
        x = min(cv2frame.shape[1] - self.half_face_detection_size, x)
        y = min(cv2frame.shape[0] - self.half_face_detection_size, y)
        img = cv2frame[int(y - self.half_face_detection_size):int(y + self.half_face_detection_size),
              int(x - self.half_face_detection_size):int(x + self.half_face_detection_size)]
        return img

    def get_face_distance(self):
        if len(self.faces) > 2:
            try:
                faces = remove_outliers(np.array(self.faces, dtype=object))
                face = np.mean(faces, axis=0)
                width = face[2] - face[0]
                height = face[3] - face[1]
                width = get_distance_width(width)
                height = get_distance_height(height)
                return np.mean(np.array([width, height]))
            except ZeroDivisionError:
                logging.error("ZeroDivisionError in get_face_distance()")
                logging.error(self.faces)

        return -1

    def run(self):
        window_name = "videostream"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        main_thread = next(
            filter(lambda t: t.name == "MainThread", threading.enumerate())
        )
        while main_thread.is_alive():
            with self.flush_queue_lock:
                try:
                    yuv_frame = self.frame_queue.get(timeout=0.01)
                except queue.Empty:
                    continue
                try:
                    # the VideoFrame.info() dictionary contains some useful information
                    # such as the video resolution
                    info = yuv_frame.info()
                    height, width = info["yuv"]["height"], info["yuv"]["width"]

                    # yuv_frame.vmeta() returns a dictionary that contains additional
                    # metadata from the drone (GPS coordinates, battery percentage, ...)

                    # convert pdraw YUV flag to OpenCV YUV flag
                    cv2_cvt_color_flag = {
                        olympe.PDRAW_YUV_FORMAT_I420: cv2.COLOR_YUV2BGR_I420,
                        olympe.PDRAW_YUV_FORMAT_NV12: cv2.COLOR_YUV2BGR_NV12,
                    }[info["yuv"]["format"]]

                    # yuv_frame.as_ndarray() is a 2D numpy array with the proper "shape"
                    # i.e (3 * height / 2, width) because it's a YUV I420 or NV12 frame

                    # Use OpenCV to convert the yuv frame to RGB
                    cv2frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv2_cvt_color_flag)
                    self.last_frame = cv2frame
                    self.show_yuv_frame(window_name, yuv_frame)
                except Exception:
                    # We have to continue popping frame from the queue even if
                    # we fail to show one frame
                    traceback.print_exc()
                finally:
                    # Don't forget to unref the yuv frame. We don't want to
                    # starve the video buffer pool
                    yuv_frame.unref()
        cv2.destroyWindow(window_name)

    def command_window_thread(self, win):
        win.nodelay(True)
        key = ""
        win.clear()
        win.addstr("Detected key:")
        while 1:
            try:
                key = win.getkey()
                win.clear()
                win.addstr("Detected key:")
                win.addstr(str(key))
                # disconnect drone
                if str(key) == "c":
                    win.clear()
                    win.addstr("c, stopping")
                    self.f.close()
                    self.pose_file.close()
                    self.drone.disconnect()
                # takeoff
                if str(key) == "t":
                    win.clear()
                    win.addstr("takeoff")
                    # assert self.drone(TakeOff()).wait().success()
                    win.addstr("completed")
                # land
                if str(key) == "l":
                    win.clear()
                    win.addstr("landing")
                    assert self.drone(Landing()).wait().success()
                # turn left
                if str(key) == "q":
                    win.clear()
                    win.addstr("turning left")
                    assert self.drone(
                        moveBy(0, 0, 0, -math.pi / 4)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    win.addstr("completed")
                # turn right
                if str(key) == "e":
                    win.clear()
                    win.addstr("turning right")
                    assert self.drone(
                        moveBy(0, 0, 0, math.pi / 4)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    win.addstr("completed")
                # move front
                if str(key) == "w":
                    win.clear()
                    win.addstr("front")
                    assert self.drone(
                        moveBy(0.2, 0, 0, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    win.addstr("completed")
                # move back
                if str(key) == "s":
                    win.clear()
                    win.addstr("back")
                    assert self.drone(
                        moveBy(-0.2, 0, 0, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    win.addstr("completed")
                # move up
                if str(key) == "r":
                    win.clear()
                    win.addstr("up")
                    assert self.drone(
                        moveBy(0, 0, -0.15, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    win.addstr("completed")
                # move down
                if str(key) == "f":
                    win.clear()
                    win.addstr("down")
                    assert self.drone(
                        moveBy(0, 0, 0.15, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    win.addstr("completed")
                # move left
                if str(key) == "a":
                    win.clear()
                    win.addstr("left")
                    assert self.drone(
                        moveBy(0, -0.2, 0, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    win.addstr("completed")
                # move right
                if str(key) == "d":
                    win.clear()
                    win.addstr("right")
                    assert self.drone(
                        moveBy(0, 0.2, 0, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    win.addstr("completed")
                # start person detection thread
                if str(key) == "p":
                    win.clear()

                    pause = self.check_for_pause()
                    if pause:
                        win.addstr("cannot start because of stop gesture")
                    else:
                        win.addstr("start detecting")
                        self.pause_finding = False
                        self.pause_finding_condition.notify()
                        self.pause_finding_condition.release()
                # pause person detecting thread
                if str(key) == "o":
                    win.clear()
                    win.addstr("stop detecting")
                    self.pause_finding = True
                    self.pause_finding_condition.acquire()
                    # self.person_thread.stop = True
                    # win.addstr("joined")
                # measure distances
                if str(key) == "x":
                    win.clear()
                    win.addstr("distances:")
                    arr = np.array(self.keypoints, dtype=object)
                    string = ""
                    for i in range(arr.shape[0]):
                        string += "{:.6f};".format(get_point_distance(arr[i], self.left_eye, self.right_eye))
                    win.addstr(string)
                    self.f.write(string + "\n")
                # measure faces
                if str(key) == "y":
                    win.clear()
                    win.addstr("distances:")
                    arr = np.array(self.faces, dtype=object)
                    width = ""
                    height = ""
                    for i in range(arr.shape[0]):
                        width += str(arr[i][2] - arr[i][0]) + ";"
                        height += str(arr[i][3] - arr[i][1]) + ";"
                    win.addstr(width + height)
                    self.f.write(width + "\n")
                    self.f.write(height + "\n")
                # log user gesture
                if str(key) == "g":
                    win.clear()
                    win.addstr("gesture made")
                    global event_time
                    event_time = time.time()
                    logging.info("stop gesture by user;{:.3f};{:.3f}".format(self.get_face_distance(), time.time()))
                # log face distances
                if str(key) == "k":
                    win.clear()
                    win.addstr("distances logged")
                    string = ""
                    arr = np.array(self.face_distances, dtype=object)
                    win.addstr(str(len(arr)))
                    for i in range(len(arr)):
                        string += "{:.2f}".format(arr[i]) + ";"
                    logging.info("distances;{}".format(string))
                    win.addstr(string)


            except Exception as e:
                # No input
                pass

    def fly_to_person(self):
        t = threading.currentThread()
        while not getattr(t, "stop", False):
            with self.pause_finding_condition:
                # wait if thread is paused
                while self.pause_finding:
                    self.pause_finding_condition.wait()
            arr = np.array(self.keypoints, dtype=object)
            if len(arr) > 2:
                pose_predictions = np.array(self.pose_predictions, dtype=object)
                if pose_predictions[-1] > 1:
                    logging.info(
                        "stop gesture {} detected;{:.3f};{:.3f}".format(pose_predictions[-1], self.get_face_distance(),
                                                                        time.time() - event_time))
                    # check if multiple stop gestures were detected
                    pause = self.check_for_pause()
                    if pause:
                        logging.info(
                            f"stopping completely gesture {pose_predictions[-1]};{self.get_face_distance()};{time.time() - event_time}")
                        # land drone
                        assert self.drone(Landing()).wait().success()
                        self.pause_finding = True
                        self.pause_finding_condition.acquire()
                    time.sleep(0.2)
                    continue
                distance = self.get_face_distance()
                xn, yn = get_point(np.average(arr[-2:], axis=0), self.nose)
                # calculate angle of nose
                angle = (xn / self.image_width - 0.5) * 1.204
                # calculate nose height in percent
                nose_height = yn / self.image_height
                # set nose to middle if none was detected
                if nose_height == 0:
                    nose_height = 0.5

                # adjust angle
                if np.abs(angle) > 0.15:
                    assert self.drone(
                        moveBy(0, 0, 0, angle)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()

                # adjust height
                elif nose_height < 0.4 and self.flight_height < self.max_height:
                    self.flight_height += 0.15
                    assert self.drone(
                        moveBy(0.0, 0, -0.15, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    time.sleep(0.4)
                elif nose_height > 0.6 and self.flight_height > 0:
                    self.flight_height -= 0.15
                    assert self.drone(
                        moveBy(0.0, 0, 0.15, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
                    time.sleep(0.4)

                # adjust distance
                elif distance > self.min_dist:
                    assert self.drone(
                        moveBy(min(0.2, distance - self.min_dist), 0, 0, 0)
                        >> FlyingStateChanged(state="hovering", _timeout=5)
                    ).wait().success()
            else:
                assert self.drone(
                    moveBy(0, 0, 0, 0.3)
                    >> FlyingStateChanged(state="hovering", _timeout=5)
                ).wait().success()

    # returns true if 4 of the 5 last pose predictions were stop
    def check_for_pause(self):
        pose_predictions = np.array(self.pose_predictions, dtype=object)
        prediction_counts = np.asarray((np.unique(pose_predictions, return_counts=True)), dtype=object).T
        for i in range(prediction_counts.shape[0]):
            if prediction_counts[i, 0] > 1 and prediction_counts[i, 1] > 3:
                return True
        return False

    # thread for keypoint predictions
    def make_predictions(self):
        while 1:
            # check if camera stream has started
            if not np.array_equal(self.last_frame, np.zeros((1, 1, 3), np.uint8)):
                # get detections
                start = time.time()
                keypoint, self.keypoints_image = self.detector.keypoint_detection(self.last_frame)
                logging.info("time for prediction = {:0.3f}".format(time.time() - start))
                # check if detection returned results
                if keypoint.size > 2:
                    self.keypoints.append(keypoint)
                    pred = self.poses_model.predict(keypoint[0].reshape(1, -1))
                    self.pose_predictions.append(int(pred[0]))
                    if pred > 1:
                        # stop moving if prediction is stop
                        logging.info("canceling move to;{:.3f};{:.3f}".format(self.get_face_distance(),
                                                                              time.time() - event_time))
                        self.drone(CancelMoveBy()).wait()
                    # put prediction on image
                    cv2.putText(self.keypoints_image, 'Classpred: ' + "{:.0f}".format(pred[0]), (0, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (36, 255, 12), 2)
                    if self.save_image:
                        cv2.imwrite("images/" + str(self.image_counter) + ".jpg", self.keypoints_image)
                        string = str(self.image_counter) + ";"
                        self.image_counter += 1
                        for i in range(keypoint.shape[1]):
                            string += "{:.2f};{:.2f};".format(keypoint[0, i, 0], keypoint[0, i, 1])
                        self.pose_file.write(string + "\n")
                # delete from last results if there is no detection
                elif len(self.keypoints) > 0:
                    self.keypoints.popleft()
                    self.pose_predictions.popleft()
                if len(self.keypoints) > 1 and self.keypoints[-1].size > 2:
                    # arr = np.array(self.keypoints, dtype=object)
                    # left_eye = remove_outliers(arr[:, 0, self.left_eye])
                    # if len(np.shape(left_eye)) > 1 and np.shape(left_eye)[0] > 1:
                    #     left_eye = np.average(left_eye, axis=0)
                    # right_eye = remove_outliers(arr[:, 0, self.right_eye])
                    # if len(np.shape(right_eye)) > 1 and np.shape(right_eye)[0] > 1:
                    #     right_eye = np.average(right_eye, axis=0)
                    # left_eye = left_eye.reshape((-1))
                    # right_eye = right_eye.reshape((-1))
                    # # eye_dist = 24.27 / np.max(
                    # #     [0.001, np.power(get_distance(np.average(arr[-2:], axis=0), self.left_eye, self.right_eye),
                    # #                      0.753)])
                    # if len(left_eye) > 1 and len(right_eye) > 1:
                    #     self.eye_dist = 24.27 / np.max(
                    #         [0.001,
                    #          np.power(math.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2),
                    #                   0.753)])

                    # display face distance
                    cv2.putText(self.keypoints_image, 'Distance(face): ' + "{:.2f}".format(self.get_face_distance()) +
                                "m", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            else:
                # wait for stream
                time.sleep(1)


def start_keyboard(streaming_example):
    curses.wrapper(streaming_example.command_window_thread)


if __name__ == "__main__":
    main_thread = MainClass()
    # Start the video stream
    main_thread.start()
    # start keyboard thread
    keyboard_thread = threading.Thread(target=start_keyboard, args=(main_thread,))
    keyboard_thread.start()
    # start predictions thread
    threading.Thread(target=main_thread.make_predictions).start()
    print("press T for takeoff")
    # wait for keyboard thread
    keyboard_thread.join()

    # Stop the video stream
    main_thread.stop()
