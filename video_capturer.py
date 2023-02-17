import cv2

import draw_helper
from dnn_box_detector_cuda import *
from dnn_pose_detector_cuda import *
from to_box_transformer import ToBoxTransformer
from after_box_transformer import AfterBoxTransformer
from to_pose_transformer import ToPoseTransformer
from cv_image_helpers import *


class VideoCapturer:
    def __init__(self, yolo_path: str, hrnet_path: str):
        self.boxDetector: DnnBoxDetectorCUDA = DnnBoxDetectorCUDA(yolo_path)
        self.poseDetector: DnnPoseDetectorCUDA = DnnPoseDetectorCUDA(hrnet_path)

        self.toBoxTransformer = ToBoxTransformer(DnnBoxDetectorCUDA.SIZE)
        self.afterBoxTransformer = AfterBoxTransformer()
        self.toPoseTransformer = ToPoseTransformer(DnnPoseDetectorCUDA.SIZE)

        self.captureHeight = -1
        self.captureWidth = -1

    def Capture(self, video_path: str):
        cap = cv2.VideoCapture(video_path)

        self.captureWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.captureHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        # length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        # fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        # boxModelPath = "models/yolov5m.onnx"
        # poseModelPath = "models/litehrnet_30_coco_384x288.onnx"

        if not cap.isOpened():
            print("Error opening video stream or file")

        while cap.isOpened():
            ret, source_img = cap.read()
            if not ret:
                break

            box_prepared_img = self.toBoxTransformer.forward(source_img)

            self.boxDetector.Forward(box_prepared_img)
            is_box_detected, box_rect = self.boxDetector.TryGetHumanBox()
            if not is_box_detected:
                continue

            after_box = self.afterBoxTransformer.Forward(
                box_prepared_img,
                size_image(box_prepared_img),
                box_rect)

            posedPrepared = self.toPoseTransformer.Forward(after_box)
            self.poseDetector.Forward(posedPrepared)

            keypoints = self.poseDetector.keypoints
            new_keypoints = []
            for kpsrc in keypoints:
                nkp = self.toPoseTransformer.Inverse(kpsrc)
                nkp = self.afterBoxTransformer.Inverse(nkp)
                nkp = self.toBoxTransformer.inverse(nkp)

                new_keypoints.append(nkp)

            final_img = draw_helper.Draw(source_img, new_keypoints)

            cv2.imshow('Frame', final_img)
            cv2.waitKey()


        cap.release()
        cv2.destroyAllWindows()
