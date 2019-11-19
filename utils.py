import cv2 as cv


def save_img(fname, img):
    cv.imwrite(fname, img)


def save_video_frame(vid_name, target_folder):
    vidcap = cv.VideoCapture(vid_name)
    success, image = vidcap.read()
    count = 0
    while success:
        dest = target_folder + f"/{count}.jpg"
        save_img(dest, image)
        success, image = vidcap.read()
        count += 1


if __name__ == "__main__":
    save_video_frame("./test_vid/test.mp4", "./test_vid_frames")
