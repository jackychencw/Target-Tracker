from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

import pathlib


class Dataset:
    def load_images(path):
        image_paths = []
        for dir_name in os.listdir(path):
            dir_path = f'{path}/{dir_name}'
            for image_name in os.listdir(dir_path):
                image_path = f'{dir_path}/{image_name}'
                image_paths.append(image_path)
        print(len(image_paths))


load_images('./image/train')
