import cv2 as cv
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-path',
                    type=str,
                    help="input path",
                    default='./people_data/Train/input/')
parser.add_argument('--mask-path',
                    type=str,
                    default='./people_data/Train/mask/',
                    help='mask path')

# parser.add_argument('--loss',
#                     type=str,
#                     default='binary_crossentropy',
#                     help='loss function')

# parser.add_argument('--save-path',
#                     type=str,
#                     default='./weights/weight.h5',
#                     help='loss function')
args = parser.parse_args()

input_path = args.input_path
mask_path = args.mask_path


def process_images(start=100, limit=600, input_path=input_path, mask_path=mask_path):
    assert os.path.exists(input_path)
    assert os.path.exists(mask_path)
    i = start
    while i <= limit:
        input_file_name = f'00{i}.jpg'
        mask_file_name = f'00{i}.png'
        input_file_path = input_path + input_file_name
        mask_file_path = mask_path + mask_file_name
        input_img = cv.imread(input_file_path)
        mask_img = cv.imread(mask_file_path)
        edges = cv.Canny(mask_img, 100, 200)
        out = np.copy(input_img)
        out[np.where(edges > 250)] = np.array([0, 255, 0])
        cv.imwrite(f'./out/{i}.jpg', out)
        i += 1


process_images()
