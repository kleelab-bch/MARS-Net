'''
Author Junbong Jang
Date 3/14/2021

preprocess cropped images by shuffling, splitting (training, validation, and test sets) and standardizing
prepare generator to lazily load them

referenced: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
'''
from data_processor import preprocess_input, preprocess_output
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
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
    print('set_training_dataset_names')
    x_filenames, y_filenames = get_cropped_filenames(round_num, dataset_names, frame, repeat_index, img_format)

    # split data
    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(
        x_filenames, y_filenames, shuffle=True, test_size=0.2, random_state=repeat_index)

    # Saving
    with open('{}x_train_filenames_{}_frame{}_repeat{}'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(x_train_filenames, fp)
    with open('{}x_val_filenames_{}_frame{}_repeat{}'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(x_val_filenames, fp)
    with open('{}y_train_filenames_{}_frame{}_repeat{}'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(y_train_filenames, fp)
    with open('{}y_val_filenames_{}_frame{}_repeat{}'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(y_val_filenames, fp)


def get_training_dataset_names(model_name, frame, repeat_index, save_path):
    print('get_training_dataset_names')
    with open('{}x_train_filenames_{}_frame{}_repeat{}'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        x_train_filenames = pickle.load(fp)
    with open('{}x_val_filenames_{}_frame{}_repeat{}'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        x_val_filenames = pickle.load(fp)
    with open('{}y_train_filenames_{}_frame{}_repeat{}'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        y_train_filenames = pickle.load(fp)
    with open('{}y_val_filenames_{}_frame{}_repeat{}'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        y_val_filenames = pickle.load(fp)

    # shuffle them
    x_train_filenames, y_train_filenames = unison_shuffle_lists(x_train_filenames, y_train_filenames)
    x_val_filenames, y_val_filenames = unison_shuffle_lists(x_val_filenames, y_val_filenames)

    return x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames


def get_dataset_generators(round_num, dataset_names, model_name, frame, repeat_index, img_format, batch_size, save_path):
    # to retrieve data in batches
    set_training_dataset_names(round_num, dataset_names, model_name, frame, repeat_index, img_format, save_path)
    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = get_training_dataset_names(model_name, frame, repeat_index, save_path)

    training_batch_generator = CustomImageGenerator(x_train_filenames, y_train_filenames, batch_size)
    validation_batch_generator = CustomImageGenerator(x_val_filenames, y_val_filenames, batch_size)

    return training_batch_generator, validation_batch_generator


class CustomImageGenerator(Sequence):
    # For loading data in batches
    # because the data is too large to fit in our memory

    def __init__(self, x_filenames, y_filenames, batch_size):
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size

        x_img = cv2.imread(x_filenames[0], cv2.IMREAD_GRAYSCALE)
        y_img = cv2.imread(y_filenames[0], cv2.IMREAD_GRAYSCALE)

        self.x_row, self.x_col = x_img.shape
        self.y_row, self.y_col = y_img.shape
        assert self.x_row == self.x_col
        assert self.y_row == self.y_col

    def __len__(self):
        return (np.ceil(len(self.x_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x_names = self.x_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_names = self.y_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_x = np.ndarray((self.batch_size, int(self.x_row), int(self.x_col)), dtype=np.uint8)
        batch_y = np.ndarray((self.batch_size, int(self.y_row), int(self.y_col)), dtype=np.uint8)

        for i, file_name in enumerate(batch_x_names):
            batch_x[i] = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        for j, file_name in enumerate(batch_y_names):
            batch_y[j] = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        # to debug image and mask mismatch 3/16/2021
        # for batch_index, x in enumerate(batch_x):
        #     x = np.moveaxis(x, 0, -1) # convert from first_channel to last_channel
        #     cv2.imwrite(f'results/history_round1_single_micro_VGG19_dropout/image_{batch_index}.png', x)
        # for batch_index, y in enumerate(batch_y):
        #     y = np.moveaxis(y, 0, -1)
        #     cv2.imwrite(f'results/history_round1_single_micro_VGG19_dropout/mask_{batch_index}.png', y)

        # standardizing or normalizing data
        batch_x = preprocess_input(batch_x)
        batch_y = preprocess_output(batch_y)

        return batch_x, batch_y

    def get_x_size(self):
        return self.x_row, self.x_col

    def get_y_size(self):
        return self.y_row, self.y_col
