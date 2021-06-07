'''
Author Junbong Jang
Date 3/14/2021

crop and augment images and save them, not as .npz files
Later, fit_generator will be used to lazy load them for training
'''


import gc
import numpy as np
import time
import cv2

import os.path
from tensorflow.keras import backend as K
import gc
import math
import sys
from tqdm import tqdm

from crop_generator import CropGenerator
from crop_generator_classifier import CropGeneratorClassifier
sys.path.append('..')
from UserParams import UserParams


def crop_dataset(round_num, dataset_name, repeat_index, input_size, output_size, img_folder, mask_folder, dataset_folder, img_format, crop_mode, crop_patches, augmentation_factor):

    if 'classifier' in constants.strategy_type:
        crop_generator = CropGeneratorClassifier(dataset_name, input_size, output_size, repeat_index, img_format, crop_mode, crop_patches, dataset_folder, img_folder, mask_folder)
    else:
        crop_generator = CropGenerator(dataset_name, input_size, output_size, repeat_index, round_num, img_format, crop_mode, crop_patches, dataset_folder, img_folder, mask_folder)

    img_train, mask_train, img_frame_names, mask_frame_names = crop_generator.crop()

    if augmentation_factor > 0:
        img_train, mask_train = crop_generator.augment_data(img_train, mask_train, repeat_index, crop_patches, augmentation_factor)

    print('img_train shape:', img_train.shape, ' mask_train shape:', mask_train.shape)

    # ---------- Saving ------------
    root_path = f'../crop/generated/crop_{crop_mode}_{dataset_name}/'
    root_path_img = root_path + f'img_repeat{repeat_index}/'
    root_path_mask = root_path + f'mask_repeat{repeat_index}/'

    if not os.path.exists(root_path_img):
        os.makedirs(root_path_img)
    if not os.path.exists(root_path_mask):
        os.makedirs(root_path_mask)

    if 'classifier' in constants.strategy_type:
        # convert mask to classes

        assert img_train.shape[0:4] == mask_train.shape[0:4]

        mask_class_dict = {}
        min_mask_area_threshold = mask_train.shape[2] * mask_train.shape[3] * 0.01
        for frame_index in tqdm(range(mask_train.shape[0])):
            for crop_index in range(mask_train.shape[1]):
                img_filepath = root_path_img + f'{img_frame_names[frame_index]}_c{crop_index}_{crop_mode}.png'
                # cv2.imwrite(img_filepath, img_train[frame_index, crop_index])

                mask_area = np.sum(mask_train[frame_index, crop_index] > 0)
                mask_class_dict[img_filepath] = mask_area > min_mask_area_threshold

        np.save(root_path_mask + 'mask_class_dict.npy', mask_class_dict)

    else:
        for frame_index in tqdm(range(img_train.shape[0])):
            for crop_index in range(img_train.shape[1]):
                cv2.imwrite(root_path_img + f'f{img_frame_names[frame_index]}_c{crop_index}_{crop_mode}.png', img_train[frame_index, crop_index])

        for frame_index in tqdm(range(mask_train.shape[0])):
            for crop_index in range(mask_train.shape[1]):
                cv2.imwrite(root_path_mask + f'f{mask_frame_names[frame_index]}_c{crop_index}_{crop_mode}.png', mask_train[frame_index, crop_index])

    print(gc.collect(), end='\n\n')  # runs garbage collection to free memory
    K.clear_session()

    return


if __name__ == "__main__":
    constants = UserParams('crop')
    print(constants)
    args = constants.get_args()
    print(args)
    for repeat_index in range(constants.REPEAT_MAX):
        for dataset_folder, img_folder, mask_folder, dataset_name in zip(constants.dataset_folders, constants.img_folders, constants.mask_folders, constants.dataset_names):
            augmentation_factor = 0
            print('@@-@@', dataset_folder, img_folder, mask_folder, dataset_name )
            crop_dataset(constants.round_num, dataset_name, repeat_index, args.input_size, args.output_size, img_folder, mask_folder, dataset_folder, constants.img_format, args.crop_mode, args.crop_patches, augmentation_factor)
