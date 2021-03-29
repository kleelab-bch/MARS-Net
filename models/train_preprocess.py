'''
Author Junbong Jang
Date 3/14/2021

preprocess cropped images by shuffling, splitting (training, validation, and test sets) and standardizing
prepare generator to lazily load them

referenced: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
'''
from data_processor import preprocess_input, preprocess_output
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

from tqdm import tqdm
import glob
import re
import random
import numpy as np
import pickle
import cv2

def unison_shuffle_lists(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a = [a[i] for i in p]
    b = [b[i] for i in p]
    return a, b


def regex_find_frame_id(filename):
    regex = re.compile(r'/f\d+_c')
    frame_id = regex.findall(filename)[0]  # example: '/f040_c'

    return frame_id


def get_cropped_filenames(round_num, dataset_names, frame, repeat_index, img_format):
    all_img_filenames = []
    all_mask_filenames = []

    for dataset_name in dataset_names:
        crop_path = '../crop/generated/crop_round{}_{}/'.format(round_num, dataset_name)
        crop_path_img = crop_path + f'img_repeat{repeat_index}/'
        crop_path_mask = crop_path + f'mask_repeat{repeat_index}/'

        img_filenames = glob.glob(crop_path_img + '*' + img_format)
        mask_filenames = glob.glob(crop_path_mask + '*' + img_format)
        assert len(img_filenames) == len(mask_filenames)

        # count unique frames
        unique_frames = []
        for img_filename in img_filenames:
            frame_id = regex_find_frame_id(img_filename)
            if frame_id not in unique_frames:
                unique_frames.append(frame_id)

        # randomly sample few frames
        sampled_frames = random.sample(unique_frames, k=frame)  # example output: ['f020', 'f010']

        # sampling img and mask filenames
        # filenames for image and mask are the same
        sampled_img_filenames = []
        for img_filename in img_filenames:
            for sampled_frame in sampled_frames:
                if sampled_frame in img_filename:
                    sampled_img_filenames.append(img_filename)
                    break
        all_img_filenames = all_img_filenames + sampled_img_filenames

        sampled_mask_filenames = []
        for mask_filename in mask_filenames:
            for sampled_frame in sampled_frames:
                if sampled_frame in mask_filename:
                    sampled_mask_filenames.append(mask_filename)
                    break
        all_mask_filenames = all_mask_filenames + sampled_mask_filenames

    return all_img_filenames, all_mask_filenames


def set_training_dataset_names(round_num, dataset_names, model_name, frame, repeat_index, img_format, save_path):
    x_filenames, y_filenames = get_cropped_filenames(round_num, dataset_names, frame, repeat_index, img_format)

    # split data
    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(
        x_filenames, y_filenames, shuffle=True, test_size=0.2, random_state=repeat_index)

    # shuffle them
    x_train_filenames, y_train_filenames = unison_shuffle_lists(x_train_filenames, y_train_filenames)
    x_val_filenames, y_val_filenames = unison_shuffle_lists(x_val_filenames, y_val_filenames)

    # Saving
    with open('{}x_train_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(x_train_filenames, fp)
    with open('{}x_val_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(x_val_filenames, fp)
    with open('{}y_train_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(y_train_filenames, fp)
    with open('{}y_val_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(y_val_filenames, fp)


def get_training_dataset_names(model_name, frame, repeat_index, save_path):
    with open('{}x_train_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        x_train_filenames = pickle.load(fp)
    with open('{}x_val_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        x_val_filenames = pickle.load(fp)
    with open('{}y_train_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        y_train_filenames = pickle.load(fp)
    with open('{}y_val_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        y_val_filenames = pickle.load(fp)

    return x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames


def get_dataset_generators(round_num, dataset_names, model_name, frame, repeat_index, img_format, batch_size, process_type, augmentation_factor, save_path):
    # to retrieve data in batches
    set_training_dataset_names(round_num, dataset_names, model_name, frame, repeat_index, img_format, save_path)
    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = get_training_dataset_names(model_name, frame, repeat_index, save_path)

    training_batch_generator = CustomImageGenerator(x_train_filenames, y_train_filenames, batch_size, process_type, repeat_index, augmentation_factor)
    validation_batch_generator = CustomImageGenerator(x_val_filenames, y_val_filenames, batch_size, process_type, repeat_index, augmentation_factor)

    return training_batch_generator, validation_batch_generator


class CustomImageGenerator(Sequence):
    # load data in batches for data that is too large to fit in RAM memory

    def __init__(self, x_filenames, y_filenames, batch_size, process_type, repeat_index, augmentation_factor):
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size
        self.process_type = process_type
        self.repeat_index = repeat_index
        self.augmentation_factor = int(augmentation_factor)

        # get image size
        x_img = cv2.imread(x_filenames[0], cv2.IMREAD_GRAYSCALE)
        y_img = cv2.imread(y_filenames[0], cv2.IMREAD_GRAYSCALE)
        self.x_row, self.x_col = x_img.shape
        self.y_row, self.y_col = y_img.shape

        assert self.x_row == self.x_col
        assert self.y_row == self.y_col

    def __len__(self):
        # return (np.ceil(len(self.x_filenames) / float(self.batch_size))).astype(np.int)
        # longer list because of augmentation

        return (np.ceil(len(self.x_filenames) * self.augmentation_factor / float(self.batch_size))).astype(np.int)

    def __getitem__(self, batch_index):
        aug_batch_index = batch_index % self.augmentation_factor
        filenames_batch_index = int(batch_index / self.augmentation_factor)
        # print('__getitem__', batch_index, aug_batch_index, filenames_batch_index)
        if aug_batch_index == 0:
            batch_x_names = self.x_filenames[
                            filenames_batch_index * self.batch_size: (filenames_batch_index + 1) * self.batch_size]
            batch_y_names = self.y_filenames[
                            filenames_batch_index * self.batch_size: (filenames_batch_index + 1) * self.batch_size]

            batch_x = np.ndarray((self.batch_size, int(self.x_row), int(self.x_col)), dtype=np.uint8)
            batch_y = np.ndarray((self.batch_size, int(self.y_row), int(self.y_col)), dtype=np.uint8)

            for i, file_name in enumerate(batch_x_names):
                batch_x[i] = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            for j, file_name in enumerate(batch_y_names):
                batch_y[j] = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

            # augmentation_factor-1 is important for keeping original images into the list
            self.augmented_batch_x, self.augmented_batch_y = self.augment_data(batch_x, batch_y, self.repeat_index,
                                                                               (self.augmentation_factor-1))
            # self.save_augmented_images(self.augmented_batch_x, self.augmented_batch_y)

            # remove channel
            self.augmented_batch_x = self.augmented_batch_x[:,0,:,:]
            self.augmented_batch_y = self.augmented_batch_y[:,0,:,:]

            # standardizing or normalizing data
            if self.process_type == 'normalize':
                self.augmented_batch_x = normalize_input(self.augmented_batch_x)
            elif self.process_type == 'standardize':
                self.augmented_batch_x = preprocess_input(self.augmented_batch_x)
            else:
                raise Exception('incorrect process_type {}'.format(self.process_type))
            self.augmented_batch_y = preprocess_output(self.augmented_batch_y)

        return self.augmented_batch_x[aug_batch_index * self.batch_size: (aug_batch_index + 1) * self.batch_size], \
               self.augmented_batch_y[aug_batch_index * self.batch_size: (aug_batch_index + 1) * self.batch_size]

    def get_x_size(self):
        return self.x_row, self.x_col

    def get_y_size(self):
        return self.y_row, self.y_col

    def augment_data(self, orig_imgs, orig_masks, repeat_index, augmentation_factor):
        datagen = ImageDataGenerator(
            rotation_range=50.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            data_format='channels_first',
            fill_mode='reflect')

        orig_imgs = orig_imgs[:, np.newaxis, :, :]
        orig_masks = orig_masks[:, np.newaxis, :, :]
        # print('Augment Data ...', orig_imgs.shape, orig_masks.shape)

        all_augmented_images = np.zeros(( (augmentation_factor+1) * self.batch_size, 1, orig_imgs.shape[2], orig_imgs.shape[3])).astype('uint8')
        all_augmented_masks = np.zeros(( (augmentation_factor+1) * self.batch_size, 1, orig_masks.shape[2], orig_masks.shape[3])).astype('uint8')

        for iteration in range(augmentation_factor):
            for augmented_images in datagen.flow(orig_imgs, batch_size=self.batch_size, seed=iteration):
                break
            for augmented_masks in datagen.flow(orig_masks, batch_size=self.batch_size, seed=iteration):
                break

            all_augmented_images[iteration*self.batch_size:(iteration+1)*self.batch_size] = augmented_images
            all_augmented_masks[iteration*self.batch_size:(iteration+1)*self.batch_size] = augmented_masks

        all_augmented_images[(iteration+1) * self.batch_size:(iteration + 2) * self.batch_size] = orig_imgs
        all_augmented_masks[(iteration+1) * self.batch_size:(iteration + 2) * self.batch_size] = orig_masks

        all_augmented_masks = all_augmented_masks[:, :, 30:orig_imgs.shape[2] - 30, 30:orig_imgs.shape[2] - 30]

        return all_augmented_images, all_augmented_masks


    def save_augmented_images(self, batch_x, batch_y):
        # to debug image and mask mismatch 3/16/2021
        all_index_list = [x for x in range(batch_x.shape[0])]
        all_index_list = all_index_list[0:100] + all_index_list[-100:]

        for batch_index in all_index_list:
            x = batch_x[batch_index]
            x = np.moveaxis(x, 0, -1) # convert from first_channel to last_channel
            cv2.imwrite(f'results/history_round1_single_micro_VGG19_dropout/image_{batch_index}.png', x)
        for batch_index in all_index_list:
            y = batch_y[batch_index]
            y = np.moveaxis(y, 0, -1)
            cv2.imwrite(f'results/history_round1_single_micro_VGG19_dropout/mask_{batch_index}.png', y)
