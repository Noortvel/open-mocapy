import os
import uuid
import logging
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.logger.setLevel(logging.INFO)

# pages

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/upload")
def upload():
    return render_template('upload.html')

@app.route("/videos/<id>")
def videos(id: str):
    return render_template('videos.html', video_id=id)

# api

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def api_upload():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        uuid_str = str(uuid.uuid4())
        new_filename = uuid_str + '_' + filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
        return {'id':new_filename}
    
    return 'Неверный файл, требуется формат *.mp4', 400


from video_capture_service import VideoCaptureService

@app.route('/api/videos/<id>/info', methods=['GET'])
def api_videos_info(id:str):
    capturer = VideoCaptureService(app.logger)
    capturer.capture(id)
    return {
        'images': capturer.skeleton_base64images,
        'keypoints': capturer.keypoints
    }
