import numpy as np

from cv_types import *
from utils import Utils
from cv_image_helpers import crop_image


class AfterBoxTransformer:
    expandValue = 30

    def __init__(self):
        self.isForwarded = False
        self.tOffset: Vec2i | None = None

    def Forward(
            self,
            boxed_img: np.ndarray,
            source_img_size: Size,
            box_rect: Rect) -> np.ndarray:

        nbox = Utils.ExpandRectVal(box_rect, AfterBoxTransformer.expandValue, source_img_size[0], source_img_size[1])
        result = crop_image(boxed_img, nbox)

        self.tOffset = Vec2i((nbox[0], nbox[1]))
        self.isForwarded = True
        return result

    def Inverse(self, point: Vec2i) -> tuple[int, int] | None:
        if not self.isForwarded:
            return None

        return (point[0] + self.tOffset[0],
                point[1] + self.tOffset[1])
