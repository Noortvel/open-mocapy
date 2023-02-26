class ProgressMonitor:
    def __init__(self, video_id):
        self.video_id = video_id
        self.max_count = 0
        self.curr_count = 0
