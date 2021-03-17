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
import psutil
import sys
from tqdm import tqdm

from crop_helpers import data_generate, expand_channel_input
sys.path.append('..')
from UserParams import UserParams


def crop_dataset(round_num, dataset_name, repeat_index, input_size, output_size, img_folder, mask_folder, dataset_folder, img_format, crop_mode, crop_patches, augmentation_factor):
    data_generator = data_generate(dataset_name, input_size, output_size, repeat_index, round_num, img_format, crop_mode, crop_patches, dataset_folder, img_folder, mask_folder)

    img_train, mask_train, frame_names = data_generator.crop()
    img_train, mask_train = data_generator.augment_data(img_train, mask_train, repeat_index, crop_patches, augmentation_factor)

    # ---------- Saving ------------
    root_path = '../crop/generated/crop_round{}_{}/'.format(round_num, dataset_name)
    root_path_img = root_path + f'img_repeat{repeat_index}/'
    root_path_mask = root_path + f'mask_repeat{repeat_index}/'

    if not os.path.exists(root_path_img):
        os.makedirs(root_path_img)
    if not os.path.exists(root_path_mask):
        os.makedirs(root_path_mask)

    # convert from first_channel to last_channel
    img_train = np.moveaxis(img_train, 1, -1)
    mask_train = np.moveaxis(mask_train, 1, -1)
    print('img_train shape:', img_train.shape, ' mask_train shape:', mask_train.shape)

    for img_index in tqdm(range(img_train.shape[0])):
        frame_index = int(img_index / (crop_patches * (augmentation_factor+1) ) )
        frame_name = frame_names[frame_index]
        crop_index = img_index % crop_patches
        aug_index = int( (img_index % (crop_patches * (augmentation_factor+1))) / crop_patches)

        cv2.imwrite(root_path_img + f'f{frame_name}_c{crop_index}_a{aug_index}.png', img_train[img_index])
        cv2.imwrite(root_path_mask + f'f{frame_name}_c{crop_index}_a{aug_index}.png', mask_train[img_index])

    # print(psutil.virtual_memory())
    print(gc.collect(), end='\n\n')  # runs garbage collection to free memory
    K.clear_session()

    return


if __name__ == "__main__":
    constants = UserParams('crop')

    args = constants.get_crop_args()
    print(args)
    for repeat_index in range(constants.REPEAT_MAX):
        for dataset_folder, img_folder, mask_folder, dataset_name in zip(constants.dataset_folders, constants.img_folders, constants.mask_folders, constants.dataset_names):
            augmentation_factor = 1 # args.augmentation_factor
            print('@@-@@', dataset_folder, img_folder, mask_folder, dataset_name )
            crop_dataset(constants.round_num, dataset_name, repeat_index, args.input_size, args.output_size, img_folder, mask_folder, dataset_folder, constants.img_format, args.crop_mode, args.crop_patches, augmentation_factor)
        gc.collect()  # runs garbage collection to free memory
