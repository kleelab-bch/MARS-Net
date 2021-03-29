'''
Author Junbong Jang
Date 3/14/2021

helper functions for crop.py
'''

import numpy as np
import os, cv2
import glob
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf

K.set_image_data_format('channels_last')

def expand_channel_input(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols, 3), dtype=np.uint8)

    for i in range(imgs.shape[0]):
        imgs_p[i, :, :, 0] = cv2.resize(imgs[i, :, :, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, :, :, 1] = cv2.resize(imgs[i, :, :, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i,:, :,  2] = cv2.resize(imgs[i, :, :, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)

    return imgs_p


class data_generate:
    def __init__(self, dataset_name, input_size, output_size, random_seed, round_num, img_format, crop_mode,
                 rand_crop_num, root, img_folder, mask_folder):
        self.dataset_name = dataset_name
        self.random_seed = random_seed
        self.round_num = round_num
        self.input_size = input_size
        self.output_size = output_size
        self.img_format = img_format
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.rand_crop_num = rand_crop_num
        self.crop_mode = crop_mode
        self.root = root

        self.row, self.col, self.total_frames = self.get_row_col()

    # ==================================================================================
    # Get the size of image and number of images
    # ==================================================================================
    def get_row_col(self):
        if self.round_num == 1:
            m_path = self.root + self.dataset_name + self.mask_folder
        else:
            m_path = self.mask_folder
        mask_list = glob.glob(m_path + '*' + self.img_format)
        img = cv2.imread(mask_list[0], cv2.IMREAD_GRAYSCALE)
        r, c = img.shape
        if self.round_num > 1:
            r = r - 30
            c = c - 30
        return float(r), float(c), len(mask_list)

    # ==================================================================================
    # Get the mask
    # ==================================================================================
    def read_mask(self, mask_f):
        mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 255
        return mask

    # ==================================================================================
    # ==================================================================================
    def r_img_mask(self):
        if self.round_num == 1:
            r_path = self.root + self.dataset_name + self.img_folder
            m_path = self.root + self.dataset_name + self.mask_folder
        else:
            r_path = self.root + self.dataset_name + self.img_folder
            m_path = self.mask_folder
        img_list = glob.glob(r_path + '*' + self.img_format)
        mask_list = glob.glob(m_path + '*' + self.img_format)

        total_number = len(mask_list)
        imgs = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        masks = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        framenames = list()

        for i in range(len(mask_list)):
            img_path = img_list[i]
            mask_path = mask_list[i]
            img_name = img_path[len(r_path):]
            mask_name = mask_path[len(r_path):]

            image_id = mask_name[-7:-4]
            img_name = img_name[:-7] + image_id + img_name[-4:]

            framenames.append(image_id)
            if self.round_num == 1:
                masks[i] = self.read_mask(mask_list[i])
                imgs[i] = cv2.imread(r_path + img_name, cv2.IMREAD_GRAYSCALE)
            elif self.round_num > 1:
                mask_orig = self.read_mask(mask_list[i])
                img_orig = cv2.imread(r_path + img_name, cv2.IMREAD_GRAYSCALE)
                row, col = img_orig.shape
                # because predicted images' border is hazy.
                imgs[i] = img_orig[30:, 30:]
                masks[i] = mask_orig[30:, 30:]

        return imgs, masks, framenames

    # ==================================================================================
    # ==================================================================================
    def sample_loc(self, edge, number, on_edge=True):
        kernel = np.ones((int(self.output_size / 2), int(self.output_size / 2)), np.uint8)
        dilate_Edge = cv2.dilate(edge, kernel, iterations=1)
        if on_edge:
            loc = np.where(dilate_Edge > 0)
        else:
            loc = np.where(dilate_Edge < 1)
        index = np.argmax([len(np.unique(loc[0])), len(np.unique(loc[1]))])
        sample_image_loc = np.random.choice(np.unique(loc[index]), number, replace=False)
        sample_pos = []
        for i in sample_image_loc:
            temp_index = np.where(loc[index] == i)[0]
            sample_pos.extend(np.random.choice(temp_index, 1))

        return loc, sample_pos

    def crop_on_loc(self, inputs, loc, sample):
        image, mask = inputs[0], inputs[1]

        imgs = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8)
        masks = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8)

        for i in range(len(sample)):
            imgs[i] = image[loc[0][sample[i]]:loc[0][sample[i]] + self.input_size,
                      loc[1][sample[i]]:loc[1][sample[i]] + self.input_size]
            masks[i] = mask[loc[0][sample[i]]:loc[0][sample[i]] + self.input_size,
                      loc[1][sample[i]]:loc[1][sample[i]] + self.input_size]
        return imgs, masks

    def crop_rand(self, inputs, edge_ratio=0.6):
        image, mask = inputs[0], inputs[1]

        edge_number = int(self.rand_crop_num * edge_ratio)
        back_number = self.rand_crop_num - edge_number

        edge = cv2.Canny(mask, 100, 200)

        loc_p, sample_p = self.sample_loc(edge, edge_number, on_edge=True)
        loc_n, sample_n = self.sample_loc(edge, back_number, on_edge=False)
        # pad and bias
        bound_in = int(np.ceil(self.input_size / 2))

        image = np.lib.pad(image, ((bound_in, bound_in), (bound_in, bound_in)), 'symmetric')
        mask = np.lib.pad(mask, ((bound_in, bound_in), (bound_in, bound_in)), 'symmetric')

        imgs_p, masks_p = self.crop_on_loc([image, mask], loc_p, sample_p)
        imgs_n, masks_n = self.crop_on_loc([image, mask], loc_n, sample_n)
        return np.r_[imgs_p, imgs_n], np.r_[masks_p, masks_n]

    # ==================================================================================
    # ==================================================================================
    def pad_img(self, inputs):
        num_y = int(np.ceil(self.col / self.output_size))
        num_x = int(np.ceil(self.row / self.output_size))
        sym = int(np.ceil(self.input_size / 2 - self.output_size / 2))
        # print('pad_img', num_y, num_x, sym)
        for i in range(3):
            inputs[i] = np.lib.pad(inputs[i], ((0, 0), (0, int(num_x * self.output_size - inputs[i].shape[1])),
                                               (0, int(num_y * self.output_size - inputs[i].shape[2]))), 'symmetric')
            inputs[i] = np.lib.pad(inputs[i], ((0, 0), (sym, sym), (sym, sym)), 'symmetric')
        return inputs[0], inputs[1], inputs[2]

    def crop_even(self, image, mask):
        image, mask = self.pad_img([image, mask])
        num_y = int(np.ceil(self.col / self.output_size))
        num_x = int(np.ceil(self.row / self.output_size))

        imgCrop = np.ndarray((self.total_frames * num_x * num_y, int(self.input_size), int(self.input_size)),
                             dtype=np.uint8)
        maskCrop = np.ndarray((self.total_frames * num_x * num_y, int(self.input_size), int(self.input_size)),
                             dtype=np.uint8)

        for row in range(num_y):
            for col in range(num_x):
                for a_frame in range(self.total_frames):
                    imgCrop[col * num_y + row] = image[a_frame,
                                                 col * self.output_size:col * self.output_size + self.input_size,
                                                 row * self.output_size:row * self.output_size + self.input_size]
                    maskCrop[col * num_y + row] = mask[a_frame,
                                                 col * self.output_size:col * self.output_size + self.input_size,
                                                 row * self.output_size:row * self.output_size + self.input_size]

        return imgCrop, maskCrop

    # ==================================================================================
    # ==================================================================================
    def crop_random(self, image, mask):
        imgs_r = np.ndarray((self.total_frames * self.rand_crop_num, int(self.input_size), int(self.input_size)),
                            dtype=np.uint8)
        masks_r = np.ndarray((self.total_frames * self.rand_crop_num, int(self.input_size), int(self.input_size)),
                            dtype=np.uint8)

        for i in range(self.total_frames):
            imgs_r[i * self.rand_crop_num:(i + 1) * self.rand_crop_num], \
            masks_r[i * self.rand_crop_num:(i + 1) * self.rand_crop_num] = self.crop_rand([image[i], mask[i]])

        return imgs_r, masks_r

    def crop(self):
        images, masks, framenames = self.r_img_mask()

        if self.crop_mode == 'random':
            imgs_train, masks_train = self.crop_random(images, masks)
        elif self.crop_mode == 'even':
            imgs_train, masks_train = self.crop_even(images, masks)
        else:
            print('Crop Mode Error:', self.crop_mode)
            exit()

        return imgs_train, masks_train, framenames

    def adjust_brightness(self, augmented_images, repeat_index):
        '''
        Not used due to brightness can make the image pixel negative or invisible
        
        '''
        for index in range(augmented_images.shape[0]):
            # change brightness
            # https://www.tensorflow.org/api_docs/python/tf/image/stateless_random_brightness
            augmented_images[index] = tf.image.stateless_random_brightness(augmented_images[index], max_delta=10,
                                                                           seed=(repeat_index, index))
    # ==================================================================================
    # ==================================================================================
    def augment_data(self, orig_imgs, orig_masks, repeat_index, crop_patches, augmentation_factor):
        datagen = ImageDataGenerator(
            rotation_range=50.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            data_format='channels_last',
            fill_mode='reflect')

        orig_imgs = orig_imgs[:, :, :, np.newaxis]
        orig_masks = orig_masks[:, :, :, np.newaxis]
        print('Augment Data ...', orig_imgs.shape, orig_masks.shape)

        for img_counter in tqdm(range(self.total_frames)):
            for iteration in range(augmentation_factor):
                for augmented_images in datagen.flow(orig_imgs[img_counter * crop_patches:(img_counter + 1) * crop_patches], batch_size=crop_patches, seed=repeat_index):
                    break
                for augmented_masks in datagen.flow(orig_masks[img_counter * crop_patches:(img_counter + 1) * crop_patches], batch_size=crop_patches, seed=repeat_index):
                    break

                if iteration > 0:
                    all_augmented_images = np.vstack([all_augmented_images, augmented_images])
                    all_augmented_masks = np.vstack([all_augmented_masks, augmented_masks])
                else:
                    all_augmented_images = augmented_images
                    all_augmented_masks = augmented_masks
                print('aug', augmented_images.shape)

            if img_counter > 0:
                imgs = np.vstack([imgs, orig_imgs[img_counter * crop_patches:(img_counter + 1) * crop_patches], all_augmented_images])
                masks = np.vstack([masks, orig_masks[img_counter * crop_patches:(img_counter + 1) * crop_patches], all_augmented_masks])
            else:
                imgs = np.vstack([orig_imgs[img_counter * crop_patches:(img_counter + 1) * crop_patches], all_augmented_images])
                masks = np.vstack([orig_masks[img_counter * crop_patches:(img_counter + 1) * crop_patches], all_augmented_masks])

            print('imgs', imgs.shape)

        masks = masks[:, 30:orig_imgs.shape[2] - 30, 30:orig_imgs.shape[2] - 30, :]

        return imgs, masks