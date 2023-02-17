import cv2
import numpy as np
from utils import Utils
from cv_image_helpers import *


class ToPoseTransformer:
    def __init__(self, pose_detector_size: (int, int)):
        self.poseDetectorSize = pose_detector_size
        self.isForwarded = False
        self.tOffset = None
        self.tScale = None

    def Forward(self, frame: np.ndarray) -> np.ndarray:

        # 1. Находим ближаший прямоугольник подгодящий по соотношению
        # сторон с детектором и добавляем размер(пустое пространство)
        f_size = size_image(frame)
        next_size = Utils.CalcNextRatioSize(f_size, self.poseDetectorSize)
        next_dt = (
            next_size[0] - f_size[0],
            next_size[1] - f_size[1])
        hbor = Utils.Cacl2Sections(next_dt[0])
        vbor = Utils.Cacl2Sections(next_dt[1])
        bordered: np.ndarray = cv2.copyMakeBorder(
            frame,
            vbor[0],
            vbor[1],
            hbor[0],
            hbor[1],
            cv2.BORDER_CONSTANT,
            (255, 255, 255))

        # 2. Изменение размера до детектора.
        out = cv2.resize(bordered, self.poseDetectorSize)
        b_size = size_image(bordered)
        o_size = size_image(out)

        self.tOffset = (hbor[0], vbor[1])
        self.tScale = (
            float(o_size[0]) / b_size[0],
            float(o_size[1]) / b_size[1])
        self.isForwarded = True

        return out

    def Inverse(self, point: Vec2i) -> Vec2i:
        nx = (point[0] / self.tScale[0]) - self.tOffset[0]
        ny = (point[1] / self.tScale[1]) - self.tOffset[1]
        return nx, ny
