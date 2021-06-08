from flask import (Flask, request, render_template, send_from_directory, url_for, jsonify)
from werkzeug.utils import secure_filename
import os
import shutil
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect.video_splitter import split_video_ffmpeg

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(basedir, './static/upload/')
app.config['ALLOWED_EXTENSIONS'] = set(['mp4', 'avi', 'ogg', 'mp3', 'mov'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def organize_folder(dirname, name):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        shutil.rmtree(full_filename)
    folder_name = os.path.join(dirname, name)
    os.makedirs(folder_name)
    return folder_name

def pyscenedetect(file, threshold, name):
    scene_list = list(0 for i in range(0,100))
    while len(scene_list) > 20:
        scene_list = [None]

        scene_dir = './static/scene/'
        folder_name = organize_folder(scene_dir, name)

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector(threshold = threshold))
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source = video_manager)

        scene_list = scene_manager.get_scene_list()

        if len(scene_list) > 20:
            threshold = threshold + 10
        else:
            scene_num = []
            for i in range (1, len(scene_list)+1):
                scene_num.append(i)
            print(scene_num)
            video_splitter = split_video_ffmpeg([video_path], scene_list, output_file_template=folder_name +'/$VIDEO_NAME-$SCENE_NUMBER.mp4', video_name=name)

            save_images(
                scene_list,
                video_manager,
                num_images = 1,
                image_name_template = '$VIDEO_NAME-$SCENE_NUMBER',
                output_dir=folder_name
            )
        start_time = []
        end_time = []
        for scene in scene_list:
            start, end = scene
            start_time.append(start.get_timecode())
            end_time.append(end.get_timecode())
    return folder_name, start_time, end_time, scene_num

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'js_static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     'static/js', filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    elif endpoint == 'css_static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     'static/css', filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route('/css/<path:filename>')
def css_static(filename):
    return send_from_directory(app.root_path + '/static/css/', filename)
@app.route('/img/<path:filename>')
def img_static(filename):
    return send_from_directory(app.root_path + '/static/img/', filename)
@app.route('/js/<path:filename>')
def js_static(filename):
    return send_from_directory(app.root_path + '/static/js/', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploadajax', methods=['POST'])
def upldfile():
    if request.method == 'POST':
        files = request.files['file']
        threshold = request.form['threshold']
        threshold = int(threshold)

        if files and allowed_file(files.filename):

            filename = secure_filename(files.filename)
            name, ext = filename.split('.')
            updir = app.config['UPLOAD_FOLDER']
            files.save(os.path.join(updir, filename))

    folder_name, start_time, end_time, scene_num = pyscenedetect(filename, threshold, name)
    file_list = os.listdir(folder_name)
    file_list_py = [file for file in file_list if file.endswith('.jpg')]
    video_list_py = [file for file in file_list if file.endswith(('mp4', 'avi', 'ogg', 'mp3', 'mov'))]

    return jsonify(name = filename, scene = file_list_py, video = video_list_py, start = start_time, end = end_time, scene_num = scene_num)

if __name__ == '__main__':
    app.run(debug=True, port=5000)