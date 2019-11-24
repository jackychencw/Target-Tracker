import cv2 as cv
import os
import glob
import numpy as np
from imutils.video import count_frames
import progressbar


def save_img(fname, img):
    cv.imwrite(fname, img)


def load_img(fname):
    img = cv.imread(fname)
    return img


def save_video_frame(vid_name, target_folder='./test_vid_frames', use_memory=True):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    assert os.path.exists(target_folder)
    num_frames = count_frames(vid_name, override=True)
    vidcap = cv.VideoCapture(vid_name)

    print("Loading frames...")
    bar = progressbar.ProgressBar(maxval=num_frames - 1,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    success, image = vidcap.read()
    if not use_memory:
        count = 0
        while success:
            dest = target_folder + f"/{count}.jpg"
            save_img(dest, image)
            success, image = vidcap.read()
            count += 1
            bar.update(count)
        print("\nDone saving images in local drive")
        return None
    else:
        image_list = [image]
        count = 0
        while success:
            success, image = vidcap.read()
            if image is not None:
                image_list.append(image)
                count += 1
                bar.update(count)
        result = image_list
        # result = np.asarray(image_list)
        print("\nDone saving image in memory")
        return result


def construct_video_from_memory(images, target_folder, vid_out):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    assert os.path.exists(target_folder)
    print("Constructing video ...")
    num_imgs = len(images)
    height, width, layers = images[0].shape
    size = (width, height)
    out_path = f'{target_folder}/{vid_out}'
    out = cv.VideoWriter(
        out_path, cv.VideoWriter_fourcc(*'DIVX'), 15, size)
    bar = progressbar.ProgressBar(maxval=num_imgs - 1,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(len(images)):
        out.write(images[i])
        bar.update(i)
    out.release()
    print("\nDone constructing video!")


def construct_video(src_folder, target_folder, vid_out):
    assert os.path.exists(src_folder)
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    assert os.path.exists(target_folder)

    path, dirs, files = next(os.walk(src_folder))
    num_frames = len(files)
    img_array = []
    for i in range(num_frames):
        fname = f'{src_folder}/{i}.jpg'
        print(fname)
        img = load_img(fname)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out_path = f'{target_folder}/{vid_out}'
    out = cv.VideoWriter(
        out_path, cv.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    vid_images = save_video_frame('./test_vid/test2.mp4')
    construct_video_from_memory(vid_images, './out', 'testproject.avi')
    # construct_video("./test_vid_processed_frames", "./out", "testproject.avi")
    None
