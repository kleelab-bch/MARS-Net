'''
Author Junbong Jang
Date 6/2/2021

Get evenly cropped FNA images for training and prediction
Includes processing and random shuffling operations
'''
from data_processor import preprocess_input, preprocess_output, normalize_input, preprocess_per_input_image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')

import tensorflow as tf
from tqdm import tqdm
import glob
import random
import numpy as np
import pickle
import cv2
from data_generator_utils import *


def get_data_generator_classifier(dataset_names, repeat_index, crop_mode, img_format, train_or_predict_mode):
    img_filenames, mask_class_dict = get_cropped_filenames_and_class_dict(dataset_names, repeat_index, crop_mode, img_format)

    img_filenames, mask_class_list = undersample_false_image_mask(img_filenames, mask_class_dict, train_or_predict_mode)

    # load data
    images = np.asarray(read_color_images(img_filenames))
    images = np.moveaxis(images, -1, 1)
    images = preprocess_input(images)
    print('Images:', images.shape, images.dtype)
    images, mask_class_list = unison_shuffle_ndarrays(images, mask_class_list)

    # split data
    if train_or_predict_mode == 'train':
        images_train, images_val, mask_class_dict_train, mask_class_dict_val = train_test_split(
            images, mask_class_list, shuffle=True, test_size=0.2, random_state=repeat_index)
        return images_train, mask_class_dict_train, images_val, mask_class_dict_val

    elif train_or_predict_mode == 'predict':
        return images, mask_class_list

    else:
        raise Exception('train_or_predict_mode is not correct', train_or_predict_mode)


def undersample_false_image_mask(img_filenames, mask_class_dict, train_or_predict_mode):
    # convert mask_class_dict to mask_class_list
    true_img_filenames = []
    false_img_filenames = []
    random.shuffle(img_filenames)

    for img_filename in img_filenames:
        if mask_class_dict[img_filename]:
            true_img_filenames.append(img_filename)
        else:
            false_img_filenames.append(img_filename)

    assert len(false_img_filenames) > len(true_img_filenames)
    max_sample_size = len(false_img_filenames)
    print('True:', len(true_img_filenames), ' False:', len(false_img_filenames))
    # undersample
    if train_or_predict_mode == 'train':
        max_sample_size = len(true_img_filenames)*10
        false_img_filenames = false_img_filenames[:max_sample_size]

    # merge true_img_filenames and false_img_filenames
    mask_class_list = np.concatenate((np.ones(len(true_img_filenames)), np.zeros(max_sample_size)), axis=0)
    all_img_filenames = np.asarray(true_img_filenames + false_img_filenames)
    all_img_filenames, mask_class_list = unison_shuffle_ndarrays(all_img_filenames, mask_class_list)
    print(all_img_filenames.shape, mask_class_list.shape)

    return all_img_filenames, mask_class_list


def get_cropped_filenames_and_class_dict(dataset_names, repeat_index, crop_mode, img_format):
    all_img_filenames = []
    all_mask_class_dict = []

    for dataset_index, dataset_name in enumerate(dataset_names):
        crop_path = f'../crop/generated/crop_{crop_mode}_{dataset_name}/'
        crop_path_img = crop_path + f'img_repeat{repeat_index}/'
        crop_path_mask = crop_path + f'mask_repeat{repeat_index}/mask_class_dict.npy'

        img_filenames = glob.glob(crop_path_img + f'*_{crop_mode}' + img_format)
        mask_class_dict = np.load(crop_path_mask, allow_pickle=True)
        all_img_filenames = all_img_filenames + img_filenames
        if dataset_index == 0:
            all_mask_class_dict = mask_class_dict.item()
        else:
            all_mask_class_dict = {**all_mask_class_dict, **mask_class_dict.item()}
    all_img_filenames = np.asarray(all_img_filenames)
    print('get_cropped_filenames_and_class_dict', len(all_img_filenames), len(all_mask_class_dict))

    return all_img_filenames, all_mask_class_dict
