'''
Author Junbong Jang
Date 3/15/2022

Predict a live cell movie using one trained model
This code is a simplified version of predict.py
'''

import sys
sys.path.append('../data_handle')

import argparse
import gc
import numpy as np
import cv2
from tqdm import tqdm
import os.path
import os
# tensorflow import must come after os.environ gpu setting
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from deep_neural_net import *
from tensorflow.keras import backend as K
from predict_data_generator import PredictDataGenerator


def prediction(weights_path, img_path, save_path, img_format):

    # ------------------- Data loading -------------------
    prediction_data_generator = PredictDataGenerator(img_path, mask_path='', strategy_type='', img_format=img_format)
    input_images, image_filenames, image_cols, image_rows, orig_cols, orig_rows = prediction_data_generator.get_expanded_whole_frames()

    print('img size:', image_rows, image_cols)
    print('orig img size:', orig_rows, orig_cols)

    # ------------------- Load the trained Model -------------------
    if "Single_Unet" in weights_path:
        model = UNet(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "Multi_VGG19D" in weights_path:
        model = VGG19_dropout(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)

    print('model layers: ', len(model.layers))

    # ------------------ Prediction ------------------------------
    segmented_output = model.predict(input_images, batch_size = 1, verbose = 1)

    save_segmented_image(segmented_output, save_path, image_filenames)

    K.clear_session()


def save_segmented_image(segmented_output, save_path, image_filenames):
    segmented_output = 255 * segmented_output  # 0=black color and 255=white color

    # save segmentation into images
    for f in tqdm(range(segmented_output.shape[0])):
        out = segmented_output[f, 0, :, :]
        cv2.imwrite(save_path + image_filenames[f], out)


if __name__ == "__main__":
    K.set_image_data_format('channels_first')

    # take user inputs from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_path", type=str, required=True)
    parser.add_argument("--live_cell_images_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--img_format", type=str, default='.png')

    args = parser.parse_args()

    trained_model_path = args.trained_model_path
    live_cell_images_path = args.live_cell_images_path
    save_path = args.save_path
    img_format = args.img_format

    print('Running predict_simple.py... Please check your trained_model_path, live_cell_images_path or save_path if the error occurs')

    prediction(trained_model_path, live_cell_images_path, save_path, img_format)
