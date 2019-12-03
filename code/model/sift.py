import matplotlib.pyplot as plt
import cv2
import numpy as np
from operator import itemgetter
from skimage.draw import line_aa

# Used opencv SIFT module to find keypoints and descriptors
# Used skimage to draw lines between points

DATA_PATH = "../input/"
OUTPUT_PATH = "../output/"


def load_color_image(filename):
    image = cv2.imread(DATA_PATH + filename)
    return image


def load_gray_scale_image(filename):
    image = cv2.imread(DATA_PATH + filename, 0)
    return image


def save_image(filename, img, path=OUTPUT_PATH):
    cv2.imwrite(path + filename, img)


def sift(img, nfeatures=100, ct=0.04, et=5, sigma=20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(
        gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, img


def sift_gray(gray, img, nfeatures=100, ct=0.04, et=5, sigma=20):
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(
        gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, img


def sift_color(img, nfeatures=100, ct=0.04, et=10, sigma=5):
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(
        img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, img


def match(kp1, des1, kp2, des2, max_n=12):
    lkp1 = len(kp1)
    lkp2 = len(kp2)
    point_list = []
    for i in range(lkp1):
        min_dis = None
        for j in range(lkp2):
            d1 = des1[i]
            d2 = des2[j]
            dis = distance(d1, d2)
            if (min_dis == None) or (dis < min_dis[2]):
                min_dis = (kp1[i], kp2[j], dis)
        point_list.append(min_dis)
    point_list.sort(key=itemgetter(2))
    return point_list[:min(len(point_list), max_n)]


def distance(des1, des2):
    # L2 Norm
    dis = des1 - des2
    l2norm = np.linalg.norm(dis, 2)

    # L1 Norm
    l1norm = np.linalg.norm(dis, 1)

    # L3 Norm
    l3norm = np.linalg.norm(dis, 3)

    return l2norm


def concatenate(i1, i2):
    assert i1.shape[0] == i2.shape[0]
    output = np.concatenate((i1, i2), axis=1)
    return output


def draw_square(img, r, c, l, color):
    y = img.shape[0]
    x = img.shape[1]
    r0 = max(r - l, 0)
    r1 = min(r + l + 1, y)
    c0 = max(c - l, 0)
    c1 = min(c + l + 1, x)
    img[r0: r1, c0: c1] = color
    return img
