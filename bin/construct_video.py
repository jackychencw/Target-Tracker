import os
import cv2 as cv
def load_img(fname):
    img = cv.imread(fname)
    return img
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
        out_path, cv.VideoWriter_fourcc(*'DIVX'), 24, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == '__main__':
    construct_video('../result', '../','hello_world.mp4')
