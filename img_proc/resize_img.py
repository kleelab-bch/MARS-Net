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


# def fix_downscale_img(a_mask_path):
#     # change height from 822 --> 821
#     img = Image.open(a_mask_path)
#     if img.size[1] == 822:
#         img = img.crop((0, 0, img.size[0], 821)) # (left, upper, right, lower)
#         img.save(a_mask_path)


def downscale_img(a_mask_path):
    basewidth = 2048
    img = Image.open(a_mask_path)
    wpercent = (basewidth/float(img.size[0]))
    hsize = math.floor(float(img.size[1])*float(wpercent))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)

    new_mask_filename = a_mask_path.replace('../assets/Spheroid/test/', f'../assets/Spheroid/test_{basewidth}/')
    new_mask_path = new_mask_filename.split('/')[:-1]
    new_mask_path = "/".join(new_mask_path)

    if os.path.isdir(new_mask_path) == 0:
        os.makedirs(new_mask_path)
    img.save(new_mask_filename)


def upscale_img(predicted_mask_path, upscale_size):
    img = Image.open(predicted_mask_path)
    img = img.resize(upscale_size, Image.ANTIALIAS)

    new_filename = predicted_mask_path.replace('predict_wholeframe_round1_spheroid_test_VGG19', 'predict_wholeframe_round1_spheroid_test_VGG19_upscaled')
    new_path = new_filename.split('/')[:-1]
    new_path = "/".join(new_path)
    if os.path.isdir(new_path) == 0:
        os.makedirs(new_path)
    img.save(new_filename)


def scale_img_spheroid():
    K.set_image_data_format('channels_first')
    constants = UserParams('predict')

    for model_index in range(len(constants.model_names)):
        dataset_folder = constants.dataset_folders[model_index]
        dataset_name = constants.dataset_names[model_index]
        model_name = constants.model_names[model_index]
        img_folder = constants.img_folders[model_index]

        # ------- downscaling original images --------
        img_path = dataset_folder + dataset_name + img_folder
        mask_path_list = glob.glob(img_path + '*' + '.png')
        for a_mask_path in tqdm(mask_path_list):
            downscale_img(a_mask_path)

        # ------- upscaling segmented images --------
        # predicted_path_list = glob.glob(f'../models/results/predict_wholeframe_round1_spheroid_test_VGG19_marsnet/{dataset_name}/frame1_train_repeat0/*.png')
        # for predicted_path in tqdm(predicted_path_list):
        #     if dataset_name in ['DMSO_2', 'FAK_2']:
        #         # 1024 x 822 --> 2048 x 1644
        #         upscale_img(predicted_path, (2048, 1644))
        #     else:
        #         # 1024 x 821 --> 4096x3286
        #         upscale_img(predicted_path, (4096, 3286))


if __name__ == "__main__":
    scale_img_spheroid()