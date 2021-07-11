'''
Author Junbong Jang
Date 6/2/2021

Prepare dataset for Multiple Task Learning
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
from data_generator_utils import *
from data_generator_utils import convert_masks_to_areas, threshold_mask_area_list


def get_data_generator_MTL(dataset_names, repeat_index, crop_mode, img_format, train_or_predict_mode):
    img_filenames, mask_filenames, mask_area_dict = get_cropped_filenames_and_class_dict(dataset_names, repeat_index, crop_mode, img_format)
    img_filenames, mask_filenames = undersample_false_image_mask(img_filenames, mask_filenames, mask_area_dict, train_or_predict_mode)

    # load images
    original_images = np.asarray(read_color_images(img_filenames))
    original_images = np.moveaxis(original_images, -1, 1)  # last channel to channel first
    images = preprocess_input(original_images)

    # load masks
    masks = np.asarray(read_images(mask_filenames))
    masks = masks[:,np.newaxis,:,:]
    masks = preprocess_output(masks)
    original_masks = masks
    masks = original_masks[:, :, 30:original_masks.shape[2] - 30, 30:original_masks.shape[2] - 30]
    # for i in range(0,50):
    #     cv2.imwrite(f'test_image{i}.png', np.moveaxis( original_images[i], 0, -1))
    #     cv2.imwrite(f'test_mask{i}.png', np.moveaxis( masks[i]*255, 0, -1))

    # images, masks, img_filenames, mask_filenames = unison_shuffle_multiple_ndarrays(images, masks, img_filenames, mask_filenames)
    print('-----------get_data_generator_MTL------------')
    print('original_images:', original_images.shape, original_images.dtype)
    print('Images:', images.shape, images.dtype)
    print('masks:', masks.shape, masks.dtype)
    threshold_mask_area_percentage = 1
    print('threshold_mask_area_percentage', threshold_mask_area_percentage)
    height, width = masks.shape[2], masks.shape[3]
    if train_or_predict_mode == 'train':
        # split data if training
        images_train, images_val, masks_train, masks_val = train_test_split(
            images, masks, shuffle=True, test_size=0.2, random_state=repeat_index)
        print('masks split', masks_train.shape, masks_val.shape)

        train_y_areas = convert_masks_to_areas(masks_train).astype(np.int32)
        valid_y_areas = convert_masks_to_areas(masks_val).astype(np.int32)
        print('train_y_areas', train_y_areas.shape, train_y_areas.dtype, train_y_areas[:100])
        train_y_classes = np.asarray(threshold_mask_area_list(height, width, train_y_areas, threshold_mask_area_percentage), dtype=np.float32)
        valid_y_classes = np.asarray(threshold_mask_area_list(height, width, valid_y_areas, threshold_mask_area_percentage), dtype=np.float32)
        print('train_y_classes', train_y_classes.shape, train_y_classes.dtype, train_y_classes[:100])
        print('True:', np.count_nonzero(train_y_classes > 0) + np.count_nonzero(valid_y_classes > 0) )
        print('False:', np.count_nonzero(train_y_classes == 0) + np.count_nonzero(valid_y_classes == 0) )
        return images_train, [masks_train, train_y_areas, train_y_classes], images_val, [masks_val, valid_y_areas, valid_y_classes]

    elif train_or_predict_mode == 'predict':
        mask_areas = convert_masks_to_areas(masks).astype(np.int32)
        print('mask_areas', mask_areas.shape, mask_areas[:10])
        mask_classes = np.asarray(threshold_mask_area_list(height, width, mask_areas, threshold_mask_area_percentage), dtype=np.float32)
        print('mask_classes', mask_classes.shape, mask_classes.dtype)
        print('True:', np.count_nonzero(mask_classes > 0) )
        print('False:', np.count_nonzero(mask_classes == 0) )

        return original_images, images, [masks, mask_areas, mask_classes], img_filenames

    elif train_or_predict_mode == 'CAM':
        return original_images, images, original_masks, img_filenames

    else:
        raise Exception('train_or_predict_mode is not correct', train_or_predict_mode)


def undersample_false_image_mask(img_filenames, mask_filenames, mask_area_dict, train_or_predict_mode):
    # convert mask_area_dict to mask_area_list
    true_img_filenames = []
    true_mask_filenames = []
    false_img_filenames = []
    false_mask_filenames = []
    img_filenames, mask_filenames = unison_shuffle_ndarrays(img_filenames, mask_filenames)

    for img_filename, mask_filename in zip(img_filenames, mask_filenames):
        if mask_area_dict[img_filename]:
            true_img_filenames.append(img_filename)
            true_mask_filenames.append(mask_filename)
        else:
            false_img_filenames.append(img_filename)
            false_mask_filenames.append(mask_filename)

    assert len(false_img_filenames) > len(true_img_filenames)
    assert len(true_img_filenames) == len(true_mask_filenames)
    print('True:', len(true_img_filenames), ' False:', len(false_img_filenames))

    # undersample
    train_undersample_ratio = 10
    max_sample_size = len(false_img_filenames)
    if train_or_predict_mode == 'train' and max_sample_size > len(true_img_filenames)*train_undersample_ratio:
        max_sample_size = len(true_img_filenames)*train_undersample_ratio
    max_sample_size = len(true_img_filenames)

    false_img_filenames = false_img_filenames[:max_sample_size]
    false_mask_filenames = false_mask_filenames[:max_sample_size]

    # merge true_img_filenames and false_img_filenames
    all_img_filenames = np.asarray(true_img_filenames + false_img_filenames)
    all_mask_filenames = np.asarray(true_mask_filenames + false_mask_filenames)

    all_img_filenames, all_mask_filenames = unison_shuffle_ndarrays(all_img_filenames, all_mask_filenames)
    print(all_img_filenames.shape, all_mask_filenames.shape)

    return all_img_filenames, all_mask_filenames


def get_cropped_filenames_and_class_dict(dataset_names, repeat_index, crop_mode, img_format):
    all_img_filenames = []
    all_mask_filenames = []
    all_mask_area_dict = []

    for dataset_index, dataset_name in enumerate(dataset_names):
        crop_path = f'../crop/generated/crop_{crop_mode}_{dataset_name}/'
        crop_path_img = crop_path + f'img_repeat{repeat_index}/'
        crop_path_mask = crop_path + f'mask_repeat{repeat_index}/'

        img_filenames = glob.glob(crop_path_img + f'*_{crop_mode}' + img_format)
        mask_filenames = glob.glob(crop_path_mask + f'*_{crop_mode}' + img_format)

        mask_area_dict = np.load(crop_path_mask + 'mask_area_dict.npy', allow_pickle=True)
        all_img_filenames = all_img_filenames + img_filenames
        all_mask_filenames = all_mask_filenames + mask_filenames

        if dataset_index == 0:
            all_mask_area_dict = mask_area_dict.item()
        else:
            all_mask_area_dict = {**all_mask_area_dict, **mask_area_dict.item()}
    all_img_filenames = np.asarray(all_img_filenames)
    all_mask_filenames = np.asarray(all_mask_filenames)
    print('get_cropped_filenames_and_class_dict', all_img_filenames.shape, all_mask_filenames.shape, len(all_mask_area_dict))
    #
    # all_img_filenames = all_img_filenames[:500]
    # all_mask_filenames = all_mask_filenames[:500]

    return all_img_filenames, all_mask_filenames, all_mask_area_dict
