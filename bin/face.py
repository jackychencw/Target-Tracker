from __future__ import absolute_import

import logging
import os
import pickle
import sys
from glob import glob

import cv2
import numpy as np

from sklearn.metrics import roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from deepface.confs.conf import DeepFaceConfs
from deepface.detectors.detector_dlib import FaceDetectorDlib
from deepface.detectors.detector_ssd import FaceDetectorSSDMobilenetV2, FaceDetectorSSDInceptionV2
from deepface.recognizers.recognizer_vgg import FaceRecognizerVGG
from deepface.recognizers.recognizer_resnet import FaceRecognizerResnet
from deepface.utils.common import get_roi, feat_distance_l2, feat_distance_cosine
from deepface.utils.visualization import draw_bboxs

logger = logging.getLogger('DeepFace')
logger.setLevel(logging.INFO if int(os.environ.get('DEBUG', 0)) == 0 else logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


class DeepFace:
    def __init__(self):
        self.detector = None
        self.recognizer = None

    def set_detector(self, detector):
        if self.detector is not None and self.detector.name() == detector:
            return
        logger.debug('set_detector old=%s new=%s' % (self.detector, detector))
        if detector == FaceDetectorDlib.NAME:
            self.detector = FaceDetectorDlib()
        elif detector == 'detector_ssd_inception_v2':
            self.detector = FaceDetectorSSDInceptionV2()
        elif detector == 'detector_ssd_mobilenet_v2':
            self.detector = FaceDetectorSSDMobilenetV2()

    def set_recognizer(self, recognizer):
        if self.recognizer is not None and self.recognizer.name() == recognizer:
            return
        logger.debug('set_recognizer old=%s new=%s' % (self.recognizer, recognizer))
        if recognizer == FaceRecognizerVGG.NAME:
            self.recognizer = FaceRecognizerVGG()
        elif recognizer == FaceRecognizerResnet.NAME:
            self.recognizer = FaceRecognizerResnet('./db.pkl')
    def recognizer_test_run(self, detector=FaceDetectorDlib.NAME, recognizer=FaceRecognizerResnet.NAME, image='./samples/ajb.jpg', visualize=False):
        self.set_detector(detector)
        self.set_recognizer(recognizer)

        if isinstance(image, str):
            logger.debug('read image, path=%s' % image)
            npimg = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            npimg = image
        else:
            logger.error('Argument image should be str or ndarray. image=%s' % str(image))
            sys.exit(-1)

        if npimg is None:
            logger.error('image can not be read, path=%s' % image)
            sys.exit(-1)

        if recognizer:
            logger.debug('run face recognition+')
            result = self.recognizer.detect([npimg[...,::-1]])
            logger.debug('run face recognition-')
        return

    def run_recognizer(self, npimg, faces, recognizer=FaceRecognizerResnet.NAME):
        self.set_recognizer(recognizer)
        rois = []
        for face in faces:
            # roi = npimg[face.y:face.y+face.h, face.x:face.x+face.w, :]
            roi = get_roi(npimg, face, roi_mode=recognizer)
            if int(os.environ.get('DEBUG_SHOW', 0)) == 1:
                cv2.imshow('roi', roi)
                cv2.waitKey(0)
            rois.append(roi)
            face.face_roi = roi

        if len(rois) > 0:
            logger.debug('run face recognition+')
            result = self.recognizer.detect(rois=rois, faces=faces)
            logger.debug('run face recognition-')
            for face_idx, face in enumerate(faces):
                face.face_feature = result['feature'][face_idx]
                logger.debug('candidates: %s' % str(result['name'][face_idx]))
                if result['name'][face_idx]:
                    name, score = result['name'][face_idx][0]
                    # if score < self.recognizer.get_threshold():
                    #     continue
                    face.face_name = name
                    face.face_score = score
        return faces

    def run(self, target_path='../result/test.jpg', detector='detector_ssd_mobilenet_v2', recognizer=FaceRecognizerResnet.NAME, image='./samples/yj/yue_jue1.jpg',
            target=None,visualize=False):
        self.set_detector(detector)
        self.set_recognizer(recognizer)

        if image is None:
            return []
        elif isinstance(image, str):
            logger.debug('read image, path=%s' % image)
            npimg = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            npimg = image
        else:
            logger.error('Argument image should be str or ndarray. image=%s' % str(image))
            sys.exit(-1)

        if npimg is None:
            logger.error('image can not be read, path=%s' % image)
            sys.exit(-1)

        logger.debug('run face detection+ %dx%d' % (npimg.shape[1], npimg.shape[0]))
        faces = self.detector.detect(npimg)
        logger.debug('run face detection- %s' % len(faces))

        if recognizer:
            faces = self.run_recognizer(npimg, faces, recognizer)
        img = draw_bboxs(np.copy(npimg), faces,target=target)
        cv2.imwrite(target_path, img)
        if visualize and visualize not in ['false', 'False']:
            cv2.imshow('DeepFace', img)
            #cv2.waitKey(0)

        return faces

    def save_and_run(self, path,image_folder_path, target_folder, target=None, visualize=True):
        """
        :param visualize:
        :param path: samples/faces
        :param image_path: samples/yue_jue1.jpg
        :param image_folder_path: source folder
        :return:
        """
        self.save_features_path(path)
        
        assert os.path.exists(image_folder_path)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        assert os.path.exists(target_folder)

        path, dirs, files = next(os.walk(image_folder_path))
        num_frames = len(files)
        img_array = []
        for i in range(num_frames):
            fname = f'{image_folder_path}/{i}.jpg'
            target_path = target_folder + f"/{i}.jpg"
            self.run(target_path, image=fname, target=target,visualize=False)
        
        

    def save_features_path(self, path="./samples/yj/faces/"):
        """

        :param path: folder contain images("./samples/faces/")
        :return:
        """
        name_paths = [(os.path.basename(img_path)[:-4], img_path)
                      for img_path in glob(os.path.join(path, "*.jpg"))]

        features = {}
        for name, path in tqdm(name_paths):
            logger.debug("finding faces for %s:" % path)
            faces = self.run(image=path,target=None)
            features[name] = faces[0].face_feature
        import pickle
        with open('db.pkl', 'wb') as f:
            pickle.dump(features, f, protocol=2)
if __name__ == '__main__':
    a = DeepFace()
    a.save_and_run('../samples/yj/faces', '../test_vid_frames', '../result',target='jacky' )

