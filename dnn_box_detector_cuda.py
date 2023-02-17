from cv_types import *
import cv2.dnn
import onnx
import onnxruntime as ort
import numpy as np


class DnnBoxDetectorCUDA:
    width = 640
    height = 640
    channels = 3
    input_tensor_size = width * height * channels
    input_node_dims = {1, channels, height, width}
    input_node_names = "images"
    output_node_names = ["output0"]  # M S {"561", "397", "458", "519"}

    personLabel = 0

    SIZE = Size((width, height))

    def __init__(self, model_path: str):
        # self.model = onnx.load_model('models/yolov5m.onnx')
        # onnx.checker.check_model(self.model)
        self.model_path = model_path
        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])  # providers=['CUDAExecutionProvider']

        self.locations: list[Vec4f] = []
        self.labels: list[int] = []
        self.confidences: list[float] = []
        self.src_rects: list[Rect] = []
        self.res_rects: list[Rect] = []
        self.res_indexs: list[int] = []
        self.recs: Rect = None
        # self.location:Vec4f = None

    def Forward(self, frame: np.ndarray):
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / float(255),
            (DnnBoxDetectorCUDA.width, DnnBoxDetectorCUDA.height),
            (0, 0, 0),
            True,
            False)

        outputs = self.ort_session.run(
            DnnBoxDetectorCUDA.output_node_names,
            {DnnBoxDetectorCUDA.input_node_names: blob})

        self.Decode(outputs[0])

    def Decode(self, output: np.ndarray):
        size = 2142000  # output_tensor[0].GetTensorTypeAndShapeInfo().GetElementCount(); // 1x25200x85=2142000
        dimensions = 85  # 0,1,2,3 ->box,4->confidence，5-85 -> coco classes confidence
        rows = size // dimensions  # 25200

        confidenceIndex = 4
        labelStartIndex = 5
        modelWidth = 640.0
        modelHeight = 640.0
        xGain = modelWidth / DnnBoxDetectorCUDA.width
        yGain = modelHeight / DnnBoxDetectorCUDA.height

        self.confidences.clear()
        self.locations.clear()
        self.src_rects.clear()
        self.labels.clear()
        self.res_indexs.clear()

        confidenceEnsure = 0.4
        confidenceEnsure2 = 0.5

        output_flat = output.flat
        for i in range(rows):
            index = i * dimensions
            if output_flat[index + confidenceIndex] <= confidenceEnsure:
                continue

            for j in range(labelStartIndex, dimensions):
                output_flat[index + j] = output_flat[index + j] * output_flat[index + confidenceIndex]

            for k in range(labelStartIndex, dimensions):
                if output_flat[index + k] <= confidenceEnsure2:
                    continue

                location = (
                    int((output_flat[index] - output_flat[index + 2] / 2) / xGain),  # top left x
                    int((output_flat[index + 1] - output_flat[index + 3] / 2) / yGain),  # top left y
                    int((output_flat[index] + output_flat[index + 2] / 2) / xGain),  # bottom right x
                    int((output_flat[index + 1] + output_flat[index + 3] / 2) / yGain)  # bottom right y
                )

                self.locations.append(location)

                rect = (location[0],
                        location[1],
                        location[2] - location[0],
                        location[3] - location[1])

                self.src_rects.append(rect)
                self.labels.append(k - labelStartIndex)
                self.confidences.append(output_flat[index + k])

    def TryGetHumanBox(self) -> (bool, Rect):
        if len(self.src_rects) == 0:
            return False, None

        if self.personLabel not in self.labels:
            return False, None

        out_rec = self.src_rects[self.personLabel]
        return True, out_rec
