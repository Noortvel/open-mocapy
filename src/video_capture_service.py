import cv2
import base64
from video_capturer import VideoCapturer
import draw_helper
from progress_monitor import ProgressMonitor


class VideoCaptureService:
    def __init__(self, logger, monitors_collection):
        self.video_root_path = 'uploads/'
        rpath = 'src/'
        yolo_path = rpath + 'models/yolov5m.onnx'
        hr_path = rpath + 'models/litehrnet_30_coco_384x288.onnx'
        self.capturer = VideoCapturer(yolo_path, hr_path, logger)
        self.skeleton_base64images = []
        self.keypoints = None
        self.monitors_collection = monitors_collection
    
    def capture(self, video_id: str):
        full_path = self.video_root_path + video_id
        self.capturer.progress_monitor = ProgressMonitor(video_id)
        self.monitors_collection[video_id] = self.capturer.progress_monitor
        self.capturer.Capture(full_path)

        for i in range(self.capturer.frames_count):
            frame = self.capturer.out_frames[i]
            keypoints = self.capturer.out_keypoints[i]
            drawed = draw_helper.Draw(frame, keypoints)
            retval, buffer = cv2.imencode('.jpg', drawed)
            jpg_as_text = base64.b64encode(buffer).decode()
            self.skeleton_base64images.append(jpg_as_text)

        self.keypoints = self.capturer.out_keypoints

