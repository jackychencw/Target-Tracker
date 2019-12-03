
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 as cv
import progressbar

from collections import defaultdict
from io import StringIO
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

SOURCE_IMAGE_PATH = '../output/out_frames'
PATH_TO_LABELS = './mscoco_complete_label_map.pbtxt'
OUTPUT_PATH = '../output/processed_frames'
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

utils_ops.tf = tf.compat.v1

tf.gfile = tf.io.gfile

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

PATH_TO_TEST_IMAGES_DIR = pathlib.Path(SOURCE_IMAGE_PATH)
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir)/"saved_model"
    print(model_dir)
    model = tf.compat.v2.saved_model.load(export_dir=str(model_dir))
    model = model.signatures['serving_default']

    return model


detection_model = load_model(model_name)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path):
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=3)
    return image_np


def process_images(image_dir=TEST_IMAGE_PATHS, out_dir=OUTPUT_PATH):
    num_images = len(TEST_IMAGE_PATHS)
    bar = progressbar.ProgressBar(maxval=num_images, widgets=[
        progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    count = 0
    for image_path in image_dir:
        image = show_inference(detection_model, image_path)
        im_rgb = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(f'{out_dir}/{count}.jpg', im_rgb)
        count += 1
        bar.update(count)


if __name__ == '__main__':
    process_images()
