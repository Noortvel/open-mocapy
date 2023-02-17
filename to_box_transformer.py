from typing import Tuple

import cv2
import numpy as np

from cv_types import *
from cv_image_helpers import crop_image


class ToBoxTransformer:
    def __init__(self, box_detector_size: Size):
        self.boxDetectorSize = box_detector_size
        self.isForwarded = False
        self.tOffset = (0, 0)
        self.tScale = (1, 1)

    def forward(self, frame: np.ndarray) -> np.ndarray:
        height: int = frame.shape[0]
        width: int = frame.shape[1]

        if width >= height:
            x_offset = (width - height) // 2
            y_offset = 0
            width_box = height
            height_box = height
        else:
            x_offset = (height - width) // 2
            y_offset = 0
            width_box = width
            height_box = height

        cropped = crop_image(frame, (x_offset, y_offset, width_box, height_box))  #frame[y_offset:y_offset + height_box, x_offset:x_offset + width_box, :]
        # print(cropped.shape)
        # cv2.imshow('Frame', cropped)
        # cv2.waitKey()

        out = cv2.resize(cropped, self.boxDetectorSize)

        self.tOffset = (x_offset, y_offset * 2)  # xOffset * 2, yOffset * 2
        self.tScale = (
                float(out.shape[1]) / cropped.shape[1],  #cols width
                float(out.shape[0]) / cropped.shape[0])  #rows height
        self.isForwarded = True
        return out

    def inverse(self, point: Vec2i) -> tuple[float, float] | None:
        if not self.isForwarded:
            return None

        new_point = (
            (point[0] / self.tScale[0]) + self.tOffset[0],
            (point[1] / self.tScale[1]) + self.tOffset[1])

        return new_point
