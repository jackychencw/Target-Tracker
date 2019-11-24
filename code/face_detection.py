import cv2 as cv
import progressbar

from utils import *


def face_detect(cascade_name, imgs):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + cascade_name)
    num_images = len(imgs)
    result = []
    print("Detecting faces...")
    bar = progressbar.ProgressBar(maxval=num_images - 1,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(num_images):
        img = imgs[i]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        result.append(img)
        bar.update(i)
    print("\nDone detecting faces")
    return result


if __name__ == "__main__":
    classifier = 'haarcascade_frontalface_alt2.xml'
    imgs = save_video_frame('./test_vid/test2.mp4')
    result = face_detect(classifier, imgs)
    construct_video_from_memory(result, './out', 'test2.avi')
