from tensorlayer.cost import dice_coe
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.models import Model, load_model
import dataset
import unet
from skimage.transform import resize
from skimage.io import imread, imshow, concatenate_images
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cwd = os.getcwd()
im_width = 128
im_height = 128
border = 5
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train-path',
                    type=str,
                    help="train path",
                    default='./people_data/Train/')
parser.add_argument('--test-path',
                    type=str,
                    default='./people_data/Test/',
                    help='test path')

parser.add_argument('--loss',
                    type=str,
                    default='binary_crossentropy',
                    help='loss function')

parser.add_argument('--save-path',
                    type=str,
                    default='./weights/weight.h5',
                    help='loss function')

parser.add_argument('--pred-path',
                    type=str,
                    default='',
                    help='pred data path')
parser.add_argument('--train',
                    type=bool, default=False, help='train or not')
parser.add_argument('--test',
                    type=bool, default=False, help='test or not')
parser.add_argument('--predict',
                    type=bool, default=False, help='predict or not')
args = parser.parse_args()

if not os.path.exists("./weights"):
    os.mkdir("./weights")
path_train = args.train_path
path_test = args.test_path
loss = args.loss
save_path = args.save_path
path_pred = args.pred_path
path_weight = save_path


def train_model(model, save_path=save_path, learning_rate=0.01, momentum=0.9, loss="binary_crossentropy", path_train=path_train):
    train_dataset = dataset.CatDataset(path_train, im_width, im_height)
    # train_dataset.augment()
    X_train, y_train = train_dataset.X, train_dataset.Y
    model.compile(optimizer=SGD(learning_rate=learning_rate,
                                momentum=momentum), loss=loss, metrics=["accuracy"])
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(save_path, monitor='accuracy', mode='max',
                        verbose=1, save_best_only=True, save_weights_only=True)
    ]
    results = model.fit(X_train, y_train, batch_size=32,
                        epochs=100, callbacks=callbacks)
    return results


def predict(weight_path, pred_path, threshold=0.2, out_path="./Output"):
    pred_dataset = dataset.CatDataset(pred_path, im_width, im_height)
    X_pred, y_pred = pred_dataset.X, pred_dataset.Y

    model.load_weights(weight_path)
    preds = model.predict(X_pred, verbose=1)
    for i in range(preds.shape[0]):
        pred = preds[i]
        print(np.where(pred > threshold))
        pred[pred > threshold] = 255.0
        pred[pred < threshold] = 0
        save_img(f'{out_path}/{i}.jpg', pred)


def test_model(weight_path, threshold, path_test=path_test):
    if not os.path.exists("./Output"):
        os.mkdir("./Output")
    if not os.path.exists("./Output/x"):
        os.mkdir("./Output/x")
    if not os.path.exists("./Output/y"):
        os.mkdir("./Output/y")
    if not os.path.exists("./Output/pred"):
        os.mkdir("./Output/pred")
    test_dataset = dataset.CatDataset(path_test, im_width, im_height)
    X_test, y_test = test_dataset.X, test_dataset.Y
    model.load_weights(weight_path)
    pred_test = model.predict(X_test, verbose=1)
    # dice_coe_result = dice_coe(pred_test, y_test, loss_type="sorensen")
    # print('Sorensen Dice Coefficient result is {}'.format(dice_coe_result))
    for i in range(pred_test.shape[0]):
        test_pred = pred_test[i]
        test_y = y_test[i] * 255
        test_x = X_test[i] * 255
        test_pred[test_pred > threshold] = 255
        test_pred[test_pred <= threshold] = 0
        save_img("./Output/x/x_{}.jpg".format(i), test_x)
        save_img("./Output/y/y_{}.jpg".format(i), test_y)
        save_img("./Output/pred/pred_{}.jpg".format(i), test_pred)

    model.compile(optimizer=SGD(), loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.evaluate(X_test, y_test)


if __name__ == "__main__":
    # # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-train", "--train", type=bool, default = True,
    #     help="Training set")
    # ap.add_argument("-test", "--test", type=bool, default=False,
    #     help="Testing set")
    # args = vars(ap.parse_args())

    # # grab the number of GPUs and store it in a conveience variable
    # train = args["train"]
    # test = args["test"]

    train = args.train
    test = args.test
    pred = args.predict
    input_img = Input((im_height, im_width, 1), name='img')
    model = unet.UNet(input_img)

    # save_path=cwd + '/weights/weight.h5'

    # loss = "binary_crossentropy"

    # Train
    if train:
        train_model(model, save_path=save_path, loss=loss)

    # Test
    if test:
        weight_path = save_path
        threshold = 0.9
        test_model(weight_path, threshold)
    if pred:
        predict(weight_path=path_weight, pred_path=path_pred)
