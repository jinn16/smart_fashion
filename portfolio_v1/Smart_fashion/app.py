from flask import (Flask, request, render_template, send_from_directory, url_for, jsonify, session)
from werkzeug.utils import secure_filename
import os, random, string, time, cv2, datetime
import shutil
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect.video_splitter import split_video_ffmpeg
from test import detect_and_color_splash, DeepFashion2Config, DeepFashion2Dataset
from lib.config import Config
from lib.model import MaskRCNN

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'static/upload/')
app.config['SPLIT_FOLDER'] = os.path.join(basedir, 'static/scene/')
app.config['SEGMENTATION_FOLDER'] = os.path.join(basedir, 'static/segmentation/')
app.config['ALLOWED_EXTENSIONS'] = set(['mp4', 'avi', 'ogg', 'mp3', 'mov'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def organize_folder():
    filenames = os.listdir(app.config['SPLIT_FOLDER'])
    full_file = []
    for filename in filenames:
        full_file.append(filename)
    random_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    while random_name in full_file:
        random_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    scene_dir = os.path.join(app.config['SPLIT_FOLDER'], random_name)
    seg_dir = os.path.join(app.config['SEGMENTATION_FOLDER'], random_name)

    os.makedirs(scene_dir)
    os.makedirs(seg_dir)

    return scene_dir, random_name

def pyscenedetect(file, threshold, name):
    scene_list = list(0 for i in range(0, 100))
    while len(scene_list) > 20:
        scene_list = [None]

        scene_dir, folder_name = organize_folder()

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        video_manager = VideoManager([video_path])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        scene_manager.add_detector(ContentDetector(threshold = threshold))
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source = video_manager)

        scene_list = scene_manager.get_scene_list()

        if 20 < len(scene_list) <= 30:
            threshold = threshold + 10
        elif 30< len(scene_list) <= 50:
            threshold = threshold + 20
        elif 50 < len(scene_list) <= 80:
            threshold = threshold + 30
        elif 80 < len(scene_list) <= 100:
            threshold = threshold + 40
        elif 100 < len(scene_list):
            threshold = threshold + 50
        else:
            video_splitter = split_video_ffmpeg([video_path], scene_list, output_file_template = scene_dir + '/$VIDEO_NAME-$SCENE_NUMBER.mp4', video_name = name)

            save_images(
                scene_list,
                video_manager,
                num_images = 1,
                image_name_template = '$VIDEO_NAME-$SCENE_NUMBER',
                output_dir = scene_dir
            )

        start_time = []
        end_time = []
        frames = []

        for scene in scene_list:
            start, end = scene
            start_time.append(start.get_timecode())
            end_time.append(end.get_timecode())

            for i, (start, end) in enumerate(scene_list):
                duration = end - start
                duration.get_frames()
                frames.append(duration.get_frames())

    return scene_dir, folder_name, start_time, end_time, frames

def object_segmentation(file_list_py, scene_dir, folder_name):

    DEFAULT_LOGS_DIR = os.path.join(basedir, "log")
    COCO_WEIGHTS_PATH = os.path.join(basedir, "weights/mask_rcnn_coco.h5")

    t = time.time()
    print("Logs: ", DEFAULT_LOGS_DIR)

    class InferenceConfig(DeepFashion2Config):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    model = MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    weights_path = 'weights/epoch_1500.h5'
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    dataset_splash = DeepFashion2Dataset()
    dataset_splash.load_coco(config.splash_img_dir, config.splash_json_path)
    dataset_splash.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset_splash.image_ids), dataset_splash.class_names))

    seg_name_list = []
    captions_list = []
    for filename in file_list_py:
        t1 = time.time()
        img = cv2.imread(os.path.join(scene_dir, filename))

        seg_name, captions = detect_and_color_splash(dataset_splash, model, image_path = filename, video_path = None, image_dir = scene_dir,
                                    folder_name = folder_name)
        seg_name_list.append(seg_name)
        captions_list.append(captions)
        print("Each iteration elapsed time : {}s".format(time.time() - t1))
    return seg_name_list, captions_list

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
        threshold = int(request.form['threshold'])

        if files and allowed_file(files.filename):
            filename = secure_filename(files.filename)
            name, ext = filename.split('.')
            updir = app.config['UPLOAD_FOLDER']
            files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    global scene_dir, folder_name
    scene_dir, folder_name, start_time, end_time, frames = pyscenedetect(filename, threshold, name)
    file_list = os.listdir(scene_dir)
    global file_list_py
    file_list_py = [file for file in file_list if file.endswith('.jpg')]
    video_list_py = [file for file in file_list if file.endswith(('mp4', 'avi', 'ogg', 'mp3', 'mov'))]

    file_list_py.sort()
    video_list_py.sort()

    return jsonify(folder_name = folder_name,scene = file_list_py, video = video_list_py, start = start_time, end = end_time, frame = frames)

@app.route('/segajax', methods=['POST'])
def segmetation():
    seg_name, captions = object_segmentation(file_list_py, scene_dir, folder_name)
    return jsonify(segmentation = seg_name, folder_name = folder_name, scene = file_list_py, captions = captions)

if __name__ == '__main__':
    app.run(debug=True, port=5000)