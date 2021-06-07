'''
Author Junbong Jang
Date 6/2/2021

helper functions for crop.py
'''

import numpy as np
import os, cv2
import glob
from tqdm import tqdm
import math

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf

K.set_image_data_format('channels_last')

class CropGeneratorClassifier:
    def __init__(self, dataset_name, input_size, output_size, random_seed, img_format, crop_mode,
                 crop_patches_num, root, img_folder, mask_folder):
        np.random.seed(random_seed)
        self.dataset_name = dataset_name
        self.input_size = input_size
        self.output_size = output_size
        self.img_format = img_format
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop_patches_num = crop_patches_num
        self.crop_mode = crop_mode
        self.root = root

        self.row, self.col, self.total_frames = self.get_row_col()

    def get_row_col(self):
        # Get the size of image and number of images
        m_path = self.root + self.dataset_name + self.mask_folder
        mask_list = glob.glob(m_path + '*' + self.img_format)
        mask = cv2.imread(mask_list[0], cv2.IMREAD_GRAYSCALE)
        r, c = mask.shape

        return float(r), float(c), len(mask_list)

    def read_mask(self, mask_f):
        mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
        mask[mask == 76] = 255  # follicular cluster
        mask[mask < 255] = 0
        return mask

    def read_img_mask(self):
        img_root_path = self.root + self.dataset_name + self.img_folder
        mask_root_path = self.root + self.dataset_name + self.mask_folder

        img_list = glob.glob(img_root_path + '*' + self.img_format)
        mask_list = glob.glob(mask_root_path + '*' + self.img_format)

        imgs = np.zeros((len(img_list), int(self.row), int(self.col), 3), dtype=np.uint8)
        masks = np.zeros((len(mask_list), int(self.row), int(self.col)), dtype=np.uint8)

        img_frame_names = []
        mask_frame_names = []

        for i in range(len(mask_list)):
            mask_path = mask_list[i]
            mask_name = mask_path[len(mask_root_path):-len(self.img_format)]

            mask_frame_names.append(mask_name)
            masks[i] = self.read_mask(mask_path)

        for i in range(len(img_list)):
            img_path = img_list[i]
            img_name = img_path[len(img_root_path):-len(self.img_format)]

            img_frame_names.append(img_name)
            imgs[i] = cv2.imread(img_path, cv2.IMREAD_COLOR)

        return imgs, masks, img_frame_names, mask_frame_names

    # ==================================================================================
    # ==================================================================================
    def pad_img(self, inputs, num_x, num_y, crop_offset):
        sym = int(np.ceil((self.input_size - self.output_size) / 2.0))
        row_expand = int( self.input_size + num_x * crop_offset - inputs.shape[1])
        col_expand = int( self.input_size + num_y * crop_offset - inputs.shape[2])
        print('pad_img', inputs.shape, sym, row_expand, col_expand, num_x, num_y)
        if inputs.ndim == 4:
            return np.lib.pad(inputs, ((0, 0), (sym, sym + row_expand), (sym, sym + col_expand), (0,0)), 'symmetric')
        elif inputs.ndim == 3:
            return np.lib.pad(inputs, ((0, 0), (sym, sym + row_expand), (sym, sym + col_expand)), 'symmetric')

    def crop_even(self, images, masks):
        # crop images evenly, considering the mask image will be 68x68 from 128x128 for training
        # The cropped images will overlap but the cropped mask images will be right next to each other at 68x68
        crop_overlap_percentage = 0  # 0.5 means 50%
        crop_offset = math.floor(self.output_size * (1 - crop_overlap_percentage))

        num_x = int(np.ceil(float(self.row - self.input_size) / crop_offset))
        num_y = int(np.ceil(float(self.col - self.input_size) / crop_offset))
        images = self.pad_img(images, num_x, num_y, crop_offset)
        masks = self.pad_img(masks, num_x, num_y, crop_offset)

        print('crop_even', images.shape, masks.shape, num_x, num_y,  self.row, self.col, crop_offset)
        imgCrop = np.zeros((images.shape[0], num_x * num_y, int(self.input_size), int(self.input_size), 3),
                             dtype=np.uint8)
        maskCrop = np.zeros((masks.shape[0], num_x * num_y, int(self.input_size), int(self.input_size)),
                             dtype=np.uint8)

        for row in range(num_y):
            for col in range(num_x):
                for a_frame in range(images.shape[0]):
                    imgCrop[a_frame, col + row * num_x] = images[a_frame,
                                                 col * crop_offset:col * crop_offset + self.input_size,
                                                 row * crop_offset:row * crop_offset + self.input_size]

                for a_frame in range(masks.shape[0]):
                    maskCrop[a_frame, col + row * num_x] = masks[a_frame,
                                                 col * crop_offset:col * crop_offset + self.input_size,
                                                 row * crop_offset:row * crop_offset + self.input_size]

        return imgCrop, maskCrop
    # ==================================================================================
    # ==================================================================================

    def crop(self):
        images, masks, img_frame_names, mask_frame_names = self.read_img_mask()
        imgs_train, masks_train = self.crop_even(images, masks)

        return imgs_train, masks_train, img_frame_names, mask_frame_names
