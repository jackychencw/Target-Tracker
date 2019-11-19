import cv2 as cv

from utils import *


def face_detect(cascade_name, src_folder, tar_folder):
    if not os.path.exists(tar_folder):
        os.mkdir(tar_folder)
    assert os.path.exists(tar_folder)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + cascade_name)

    for fname in os.listdir(src_folder):
        src_file = f'{src_folder}/{fname}'
        tar_file = f'{tar_folder}/{fname}'
        img = load_img(src_file)
        if img is not None:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            save_img(tar_file, img)


if __name__ == "__main__":
    classifier = 'haarcascade_frontalface_default.xml'
    face_detect(classifier, './test_vid_frames', './test_vid_processed_frames')
