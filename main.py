import cv2
import numpy as np
from video_capturer import VideoCapturer

video_capturer = VideoCapturer('models/yolov5m.onnx', 'models/litehrnet_30_coco_384x288.onnx')
video_capturer.Capture('example_videos/movie_001.mp4')

cv2.destroyAllWindows()

# 'movie_001.mp4'
# cap = cv2.VideoCapture('movie_001.mp4')

# length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
# width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
# fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)

# boxModelPath = "models/yolov5m.onnx"
# poseModelPath = "models/litehrnet_30_coco_384x288.onnx"


# if not cap.isOpened():
#     print("Error opening video stream or file")
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if ret:
#         # cv2.imshow('Frame', frame)
#         # print(frame.shape)
#         tframe: np.ndarray = frame[0:720, 0:720, :]
#         print(tframe.shape)
#         cv2.imshow('Frame', tframe)
#         cv2.waitKey()
#     else:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
