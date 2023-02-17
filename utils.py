from cv_types import *


class Utils:

    @staticmethod
    def CalcExpand(value: float) -> float:
        _eps = 0.00001
        half = value / 2.0
        r_half = int(half)
        offset = 1 if (half - r_half) > _eps else 0
        result_expand = r_half + offset
        return result_expand

    @staticmethod
    def Cacl2Sections(value: float) -> (int, int):
        _eps = 0.00001
        half = value / 2.0
        r_half = int(half)
        offset = 1 if (half - r_half) > _eps else 0
        result_expand = r_half + offset

        return r_half, result_expand

    @staticmethod
    def CalcNextRatioSize(original: (int, int), destination: (int, int)) -> (int, int):
        target_ratio = destination[1] / float(destination[0])
        original_ratio = original[1] / float(original[0])
        scale = target_ratio * float(original[0]) / float(original[1])
        new_width = original[0]
        new_height = original[1]

        if original_ratio < target_ratio:
            new_height = float(original[1]) * scale
        else:
            new_width = float(original[0]) / scale

        return int(new_width), int(new_height)

    @staticmethod
    def ExpandRectVal(
            src: Rect,
            value: float,
            max_width: float,
            max_height: float):
        return Utils.ExpandRect(src, value, value, max_width, max_height)

    @staticmethod
    def ExpandRect(
            src: Rect,
            width_value: float,
            height_value: float,
            max_width: float,
            max_height: float) -> Rect:

        width_expand = Utils.CalcExpand(width_value)
        height_expand = Utils.CalcExpand(height_value)

        r_x = src[0] - width_expand
        r_y = src[1] - height_expand
        r_width = src[2] + width_value
        r_height = src[3] + height_value

        if r_x < 0:
            r_x = 0
        if r_y < 0:
            r_y = 0
        if r_width > max_width:
            r_width = max_width
        if r_height > max_height:
            r_height = max_height

        return r_x, r_y, r_width, r_height
