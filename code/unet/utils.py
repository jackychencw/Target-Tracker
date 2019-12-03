import cv2 as cv 
import matplotlib.pyplot as plt
import os
import argparse


TRAIN_DATA_INPUT_PATH = "./cat_data/Train/input/"
TRAIN_DATA_MASK_PATH = "./cat_data/Train/mask/"
OUTPUT_PATH = "./Output/"
IMAGE_W = 128
IMAGE_H = 128

def load_color_image(filepath):
    image = cv.imread(filepath)
    return image

def load_grey_scale_image(filepath):
    image = cv.imread(filepath, 0)
    return image

def show_image(img):
    cv.imshow("Showing image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()