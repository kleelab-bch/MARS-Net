'''
Author Junbong Jang
Date: 7/23/2021

# resize images for Spheroid project
'''

import sys
sys.path.append('..')
from UserParams import UserParams

import os
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras import backend as K
import glob
from tqdm import tqdm
import math


# def fix_downscale_img(a_img_path):
#     # change height from 822 --> 821
#     img = Image.open(a_img_path)
#     if img.size[1] == 822:
#         img = img.crop((0, 0, img.size[0], 821)) # (left, upper, right, lower)
#         img.save(a_img_path)


def downscale_img(a_img_path, target_size):
    img = Image.open(a_img_path)
    if type(target_size) is tuple:
        img = img.resize(target_size, Image.ANTIALIAS)
        new_img_filename = a_img_path.replace('../assets/Spheroid/test/', f'../assets/Spheroid/test_{target_size[0]}_{target_size[1]}/')
    else:
        wpercent = (target_size/float(img.size[0]))
        hsize = math.floor(float(img.size[1])*float(wpercent))
        target_size = (target_size, hsize)
        img = img.resize(target_size, Image.ANTIALIAS)
        new_img_filename = a_img_path.replace('../assets/Spheroid/test/', f'../assets/Spheroid/test_{target_size[0]}/')
    new_img_path = new_img_filename.split('/')[:-1]
    new_img_path = "/".join(new_img_path)

    if os.path.isdir(new_img_path) == 0:
        os.makedirs(new_img_path)
    img.save(new_img_filename)


def upscale_img(strategy_type, predicted_img_path, upscale_size):
    img = Image.open(predicted_img_path)
    img = img.resize(upscale_size, Image.ANTIALIAS)

    new_filename = predicted_img_path.replace(f'predict_wholeframe_round1_{strategy_type}', f'predict_wholeframe_round1_{strategy_type}_upscaled')
    new_path = new_filename.split('/')[:-1]
    new_path = "/".join(new_path)
    if os.path.isdir(new_path) == 0:
        os.makedirs(new_path)
    img.save(new_filename)


def rescale_img(img_path):
    img = Image.open(img_path)
    img = np.array(img)

    img_mean = np.mean(img)
    img_std = np.std(img)
    min = math.ceil(img_mean - img_std * 2)
    max = math.floor(img_mean + img_std * 3)
    img[img < min] = min
    img[img > max] = max

    return img


def rescale_test_img(img_path, target_size):
    # rescale each image's pixel intensity for Spheroid project
    print(img_path)
    img = rescale_img(img_path)

    img = Image.fromarray(img)

    if type(target_size) is tuple:
        new_filename = img_path.replace(f'../assets/Spheroid/test_{target_size[0]}_{target_size[1]}/',
                                        f'../assets/Spheroid/test_{target_size[0]}_{target_size[1]}_rescaled/')
    else:
        new_filename = img_path.replace(f'../assets/Spheroid/test_{target_size}/',
                                        f'../assets/Spheroid/test_{target_size}_rescaled/')
    print(new_filename)
    new_path = new_filename.split('/')[:-1]
    new_path = "/".join(new_path)
    if os.path.isdir(new_path) == 0:
        os.makedirs(new_path)
    # img.save(new_filename)


def rescale_train_img(img_path):
    # rescale each image's pixel intensity for Spheroid project
    img = rescale_img(img_path)

    img = Image.fromarray(img)
    new_filename = img_path.replace(f'/img/', f'/img_rescaled/')
    new_path = new_filename.split('/')[:-1]
    new_path = "/".join(new_path)
    if os.path.isdir(new_path) == 0:
        os.makedirs(new_path)
    img.save(new_filename)


def preprocess_img_spheroid(mode, preprocess_type, target_size):
    print('preprocess_img_spheroid', mode, preprocess_type, target_size)
    K.set_image_data_format('channels_first')
    constants = UserParams(mode)

    for model_index in range(len(constants.model_names)):
        print('model_index', model_index)
        dataset_folder = constants.dataset_folders[model_index]
        dataset_name = constants.dataset_names[model_index]
        model_name = constants.model_names[model_index]
        img_folder = constants.img_folders[model_index]

        # ------- downscaling original images --------
        if preprocess_type == 'dowscale_size_orig':
            img_path = dataset_folder + dataset_name + img_folder
            img_path_list = glob.glob(img_path + '*' + '.png')
            for a_img_path in tqdm(img_path_list):
                downscale_img(a_img_path, target_size)

        # ------- upscaling segmented images --------
        elif preprocess_type == 'upscale_size_segmented':
            predicted_path_list = glob.glob(f'../models/results/predict_wholeframe_round1_{constants.strategy_type}/{dataset_name}/frame1_train_repeat0/*.png')
            for predicted_path in tqdm(predicted_path_list):
                if dataset_name in ['DMSO_2', 'FAK_2']:
                    # 1024 x 822 --> 2048 x 1644
                    upscale_img(constants.strategy_type, predicted_path, (2048, 1644))
                else:
                    # 1024 x 821 --> 4096x3286
                    upscale_img(constants.strategy_type, predicted_path, (4096, 3286))

        # ------- rescale training images --------
        elif preprocess_type == 'rescale_pixel' and mode == 'train':
            img_path = dataset_folder + dataset_name + img_folder
            img_path_list = glob.glob(img_path + '*' + '.png')
            for a_img_path in tqdm(img_path_list):
                rescale_train_img(a_img_path)

        # ------- rescale test images --------
        elif preprocess_type == 'rescale_pixel' and mode == 'predict':
            img_path = dataset_folder + dataset_name + img_folder
            img_path_list = glob.glob(img_path + '*' + '.png')
            for a_img_path in tqdm(img_path_list):
                rescale_test_img(a_img_path, target_size)

        else:
            print('Preprocess_type {preprocess_type} is not Found')


if __name__ == "__main__":
    target_size = 1024
    # target_size = (1182, 832)
    print(target_size, type(target_size) is tuple)

    # preprocess_img_spheroid('train','dowscale_size_orig', target_size)
    # preprocess_img_spheroid('train','rescale_pixel', target_size)

    # preprocess_img_spheroid('predict','dowscale_size_orig', target_size)
    preprocess_img_spheroid('predict','rescale_pixel', target_size)

    # preprocess_img_spheroid('predict','upscale_size_segmented', target_size)