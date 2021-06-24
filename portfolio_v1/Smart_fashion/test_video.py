# -*- coding: utf-8 -*-

import sys
import os
import time

import skimage.draw
import datetime

sys.dont_write_bytecode = True

import numpy as np
import cv2

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from lib.config import Config
from lib.model import MaskRCNN
from lib import utils
from lib import visualize


class DeepFashion2Config(Config):

    # Give the configuration a recognizable name
    NAME = "deepfashion2"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    GPU_COUNT = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # COCO has 80 classes

    USE_MINI_MASK = True

    splash_img_dir = "/home1/Dataset/Deepfashion2/train/image"
    splash_json_path = "/home/jaehwang/workspace/lincon/gitref/match_rcnn/json_files/splash.json"


############################################################
#  Dataset
############################################################

class DeepFashion2Dataset(utils.Dataset):
    def load_coco(self, image_dir, json_path, class_ids=None,
                  class_map=None, return_coco=False):
        """Load the DeepFashion2 dataset.
        """

        coco = COCO(json_path)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("deepfashion2", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "deepfashion2", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 127
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(dataset, model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(filename))
        # Read image
        image = skimage.io.imread(os.path.join(IMAGE_DIR,filename))
        # Detect objects
        r = model.detect([image], verbose=1)[0]

        # Save each segmented masked images for searching items from a classified image DB.
        mask = r['masks']
        mask = mask.astype(int)
        mask.shape

        for i in range(mask.shape[2]):
            temp = skimage.io.imread(os.path.join(IMAGE_DIR,filename))
            for j in range(temp.shape[2]):
                temp[:, :, j] = temp[:, :, j] * mask[:, :, i]

        # proposal classification
        mrcnn = model.run_graph([image], [
            ("proposals", model.keras_model.get_layer("ROI").output),
            ("probs", model.keras_model.get_layer("mrcnn_class").output),
            ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
            ("masks", model.keras_model.get_layer("mrcnn_mask").output),
            ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ])

        # Get detection class IDs. Trim zero padding.
        det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
        det_count = np.where(det_class_ids == 0)[0][0]
        det_class_ids = det_class_ids[:det_count]
        detections = mrcnn['detections'][0, :det_count]

        print("{} detections: {}".format(
            det_count, np.array(dataset.class_names)[det_class_ids]))

        captions = ["{}: {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
                    for c, s in zip(detections[:, 4], detections[:, 5])]
        print(captions)

        file_name = "segs_bbox_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
        visualize.save_image(image, file_name, r['rois'], r['masks'], r['class_ids'], r['scores'], dataset.class_names, scores_thresh=0.8, save_dir= 'test_output', mode=0)

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("/home/jaehwang/workspace/lincon/gitref/match_rcnn/")
    DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "log")
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Match R-CNN for DeepFashion.')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    t = time.time()

    print("Logs: ", DEFAULT_LOGS_DIR)

    class InferenceConfig(DeepFashion2Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    weights_path = 'weights/epoch_1500.h5'

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    dataset_splash = DeepFashion2Dataset()
    dataset_splash.load_coco(config.splash_img_dir, config.splash_json_path)
    dataset_splash.prepare()
    print("Images: {}\nClasses: {}".format(len(dataset_splash.image_ids), dataset_splash.class_names))

    detect_and_color_splash(dataset_splash, model, image_path=None, video_path=args.video)
