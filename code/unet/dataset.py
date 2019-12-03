import torch
import os
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import numpy as np
from skimage.transform import resize


class CatDataset(Dataset):
    def __init__(self, rootdirectory, im_width=128, im_height=128):
        input_directory = rootdirectory + 'input/'
        mask_directory = rootdirectory + 'mask/'
        self.len = len(os.listdir(input_directory))
        self.X = np.zeros((self.len, im_height, im_width, 1), dtype=np.float32)
        self.Y = np.zeros((self.len, im_height, im_width, 1), dtype=np.float32)

        id = 0
        for inputfilename in os.listdir(input_directory):
            file_path = input_directory + inputfilename
            input_img = load_img(file_path, color_mode="grayscale")
            if input_img is not None:
                input_img = img_to_array(input_img)
                input_img = resize(input_img, (int(im_height), int(im_width)))
                new_input_img = torch.from_numpy(input_img)
                self.X[id, ..., 0] = new_input_img.squeeze() / 255
            id += 1
        id = 0
        if os.path.exists(mask_directory) and len(os.listdir(mask_directory)) != 0:
            for maskfilename in os.listdir(mask_directory):
                file_path = mask_directory + maskfilename
                mask_img = load_img(file_path, color_mode="grayscale")
                if mask_img is not None:
                    mask_img = img_to_array(mask_img)
                    mask_img = resize(
                        mask_img, (int(im_height), int(im_width)))
                    new_mask_img = torch.from_numpy(mask_img)
                    self.Y[id] = new_mask_img / 255
                id += 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

    def augment(self):
        data_size = self.X.shape
        x = self.X
        y = self.Y

        # flip
        axis = np.random.randint(1, 2)
        flipped_x = np.flip(x, axis=axis)
        flipped_y = np.flip(y, axis=axis)

        # noise
        noise = np.random.randint(5, size=data_size, dtype='uint8')
        noised_x = x + noise
        noised_y = y

        # rotate
        rotated_x = np.zeros(data_size)
        rotated_y = np.zeros(data_size)

        croped_x = np.zeros(data_size)
        croped_y = np.zeros(data_size)
        for i in range(data_size[0]):
            _x = self.X[i]
            t = self.Y[i]

            k = np.random.randint(1, 3)
            rotated_x[i] = np.rot90(_x, k)
            rotated_y[i] = np.rot90(t, k)

            hd = np.random.randint(0, 1)
            wd = np.random.randint(0, 1)
            h_crop = np.random.randint(0, data_size[1]/5)
            w_crop = np.random.randint(0, data_size[2]/5)
            # crop from top if hd = 0, else from bottom
            if hd == 0:
                # crop from left if wd == 0, else from right
                if wd == 0:
                    cx = _x[h_crop:, w_crop:]
                    ct = t[h_crop:, w_crop:]
                else:
                    cx = _x[h_crop:, :-w_crop]
                    ct = t[h_crop:, :-w_crop]
            else:
                if wd == 0:
                    cx = _x[:-h_crop, w_crop:]
                    ct = t[:-h_crop, w_crop:]
                else:
                    cx = _x[:-h_crop, :-w_crop]
                    ct = t[:-h_crop, :-w_crop]
            croped_x[i] = resize(cx, (data_size[1], data_size[2]))
            croped_y[i] = resize(ct, (data_size[1], data_size[2]))
        aug_x = np.concatenate(
            (x, flipped_x, noised_x, rotated_x, croped_x), axis=0)
        aug_y = np.concatenate(
            (y, flipped_y, noised_y, rotated_y, croped_y), axis=0)

        self.X = aug_x
        self.Y = aug_y
