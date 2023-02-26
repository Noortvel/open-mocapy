import numpy as np
import cv2
from cv_types import *


def Draw(img: np.ndarray, keypoints: list[Vec2i]) -> np.ndarray:
    show_points = []
    for kp in keypoints:
        x = round(kp[0])
        y = round(kp[1])
        show_points.append(cv2.KeyPoint(x, y, 2))

    out_img = cv2.drawKeypoints(img, show_points, None, (0, 0, 255), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    return out_img

def DrawSkeleton(img: np.ndarray, keypoints: list[Vec2i]) -> np.ndarray:
    show_points = []
    for kp in keypoints:
        x = round(kp[0])
        y = round(kp[1])
        show_points.append(cv2.KeyPoint(x, y, 2))

    out_img = cv2.drawKeypoints(img, show_points, None, (0, 0, 255), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    return out_img
