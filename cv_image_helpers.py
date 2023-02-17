import numpy as np
from cv_types import *


def crop_image(img: np.ndarray, rect: Rect) -> np.ndarray:
    return img[rect[1]:rect[3] + rect[1], rect[0]:rect[2] + rect[0], :]


def size_image(img: np.ndarray) -> Size:
    return img.shape[1], img.shape[0]
