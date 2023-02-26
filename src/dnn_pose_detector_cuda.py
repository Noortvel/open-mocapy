import cv2
import numpy as np
import cv2.dnn
import onnxruntime as ort

from cv_types import *


class DnnPoseDetectorCUDA:
    width = 288
    height = 384
    channels = 3
    batch = 1
    input_tensor_size = batch * channels * height * width
    input_node_dims = (batch, channels, height, width)
    input_node_names = "input.1"
    output_node_names = ["8674"]

    KEYPOINTS_COUNT = 17
    INPUT_SIZE: Size = (width, height)
    OUT_SIZE: Size = Size((72, 96))
    NORMALIZE_MEAN: Vec3f = Vec3f((0.485, 0.456, 0.406))
    NORMALIZE_STD: Vec3f = (0.229, 0.224, 0.225)

    SIZE = Size((288, 384))

    def __init__(self, model_path: str):
        self.keypoints: list[Vec2i] = []
        for i in range(DnnPoseDetectorCUDA.KEYPOINTS_COUNT):
            self.keypoints.append((0, 0))
        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def NormalizeMeanStd(self, img: np.ndarray):
        meanNrm, stdNrm = cv2.meanStdDev(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    img[i, j, k] -= meanNrm[k]
                    img[i, j, k] /= stdNrm[k]
                    img[i, j, k] *= DnnPoseDetectorCUDA.NORMALIZE_STD[k]
                    img[i, j, k] += DnnPoseDetectorCUDA.NORMALIZE_MEAN[k]

        return img

    def HeatmapToKeypoints(self, img: np.ndarray):
        # TODO: find in blob directly
        img_flat = img.flat
        hm = np.zeros((DnnPoseDetectorCUDA.OUT_SIZE[1], DnnPoseDetectorCUDA.OUT_SIZE[0]))

        for layer in range(DnnPoseDetectorCUDA.KEYPOINTS_COUNT):
            for y in range(DnnPoseDetectorCUDA.OUT_SIZE[1]):
                for x in range(DnnPoseDetectorCUDA.OUT_SIZE[0]):
                    indx = layer * DnnPoseDetectorCUDA.OUT_SIZE[0] * DnnPoseDetectorCUDA.OUT_SIZE[1] + y * DnnPoseDetectorCUDA.OUT_SIZE[0] + x
                    src_val = img_flat[indx]
                    hm[y, x] = src_val

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hm)
            nx = (max_loc[0] / DnnPoseDetectorCUDA.OUT_SIZE[0]) * DnnPoseDetectorCUDA.INPUT_SIZE[0]
            nx = round(nx)
            ny = (max_loc[1] / DnnPoseDetectorCUDA.OUT_SIZE[1]) * DnnPoseDetectorCUDA.INPUT_SIZE[1]
            ny = round(ny)

            self.keypoints[layer] = (nx, ny)

    def FitToSourceImage(self, img: np.ndarray):
        for i in range(len(self.keypoints)):
            kp = self.keypoints[i]
            nx = (kp[0] / DnnPoseDetectorCUDA.INPUT_SIZE[0]) * img.shape[1]
            nx = round(nx)
            ny = kp[1] / DnnPoseDetectorCUDA.INPUT_SIZE[1] * img.shape[0]
            ny = round(ny)

            self.keypoints[i] = (nx, ny)

    def Forward(self, source: np.ndarray) -> list[Vec2i]:
        img = np.float32(source)
        cv2.resize(img, DnnPoseDetectorCUDA.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.NormalizeMeanStd(img)

        blob = cv2.dnn.blobFromImage(img)
        outputs = self.ort_session.run(
            DnnPoseDetectorCUDA.output_node_names,
            {DnnPoseDetectorCUDA.input_node_names: blob})
        result = outputs[0]

        self.HeatmapToKeypoints(result)
        self.FitToSourceImage(source)

        return self.keypoints

