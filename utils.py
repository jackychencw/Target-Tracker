import cv2 as cv
import os
import glob


def save_img(fname, img):
    cv.imwrite(fname, img)


def load_img(fname):
    print(fname)
    img = cv.imread(fname)
    return img


def save_video_frame(vid_name, target_folder):
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    assert os.path.exists(target_folder)

    vidcap = cv.VideoCapture(vid_name)
    success, image = vidcap.read()
    count = 0
    while success:
        dest = target_folder + f"/{count}.jpg"
        save_img(dest, image)
        success, image = vidcap.read()
        count += 1


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
    # save_video_frame('./test_vid/test.mp4', './test_vid_frames')
    # construct_video("./test_vid_frames", "./out", "project.avi")
