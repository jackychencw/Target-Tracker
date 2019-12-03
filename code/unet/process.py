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
    for filename in os.listdir(input_path):
        input_file_path = input_path + filename
        mask_file_path = mask_path + filename
        input_img = cv.imread(input_file_path)
        mask_img = cv.imread(mask_file_path)
        mask_img = cv.resize(
            mask_img, (input_img.shape[1], input_img.shape[0]))
        print(input_img.shape)
        print(mask_img.shape)
        edges = cv.Canny(mask_img, 100, 200)
        out = np.copy(input_img)
        out[np.where(edges > 250)] = np.array([0, 255, 0])
        cv.imwrite(f'./out/{filename}', out)


process_images()
