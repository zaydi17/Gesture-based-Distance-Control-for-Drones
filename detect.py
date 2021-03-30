import threading
import imutils
import numpy as np
import cv2



from imutils.object_detection import non_max_suppression

import mxnet as mx
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints


class Detector(threading.Thread):
    def __init__(self):

        # set constants
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


        # init dnn face detection
        modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        # modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt.txt"
        self.face_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

        #pose detection
        self.ctx = mx.cpu()
        self.person_detector = get_model("ssd_512_mobilenet1.0_coco", pretrained=True, ctx=self.ctx)
        self.person_detector.reset_class(classes=['person'], reuse_weights={'person': 'person'})
        self.pose_estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=self.ctx)


    def keypoint_detection(self,frame ):
        img = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(img, short=480, max_size=1024)
        x = x.as_in_context(self.ctx)
        class_IDs, scores, bounding_boxs = self.person_detector(x)
        pred_coords = np.zeros(1)
        pose_input, upscale_bbox = detector_to_simple_pose(scaled_img, class_IDs, scores, bounding_boxs,
                                                           output_shape=(128, 96), ctx=self.ctx)
        if len(upscale_bbox) > 0:
            predicted_heatmap = self.pose_estimator(pose_input)
            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
            scale = 1.0 * img.shape[0] / scaled_img.shape[0]
            img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                    box_thresh=1, keypoint_thresh=0.3, scale=scale)
            pred_coords *= scale


        if isinstance(img, mx.nd.NDArray):
            img = frame
        if isinstance(pred_coords, mx.nd.NDArray):
            pred_coords = pred_coords.asnumpy()
        return pred_coords, img


    def detect_face(self, frame):
        img = frame
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 117.0, 123.0))
        self.face_net.setInput(blob)
        faces = self.face_net.forward()[0,0]

        face = np.array([])
        # get face with highest confidence
        faces = faces[faces[:,2].argsort()]#np.sort(faces,axis=2)
        i = faces.shape[0]-1
        confidence = faces[i, 2]
        if confidence > 0.5:
            box = faces[i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
            face = np.array([x,y,x1,y1])
        return face, img