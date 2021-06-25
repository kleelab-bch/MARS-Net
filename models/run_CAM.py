'''
Junbong Jang
6/7/2021

CAM for image classifier
'''
import sys
sys.path.append('..')
sys.path.append('../visualization')

from CAM import visualize_feature_activation_map

import os

from tensorflow.keras import backend as K
import cv2
import numpy as np

from predict_data_generator import PredictDataGenerator
from data_processor import preprocess_input
from data_generator_classifier import get_data_generator_classifier
from data_generator_MTL import get_data_generator_MTL
from data_generator_utils import convert_masks_to_classes
from model_builder import build_model_predict
from UserParams import UserParams
from tqdm import tqdm


if __name__ == "__main__":
    K.set_image_data_format('channels_first')
    constants = UserParams('predict')

    if len(constants.model_names) != 1 and len(constants.dataset_names) != len(constants.model_names):
        raise Exception('Length of Dataset names and Model names are not the same')

    for repeat_index in range(constants.REPEAT_MAX):
        for frame in constants.frame_list:
            for model_index in range(len(constants.model_names)):

                model_name = constants.model_names[model_index]
                dataset_folder = constants.dataset_folders[model_index]
                dataset_name = constants.dataset_names[model_index]
                img_folder = constants.img_folders[model_index]
                mask_folder = constants.mask_folders[model_index]

                img_path = dataset_folder + dataset_name + img_folder
                mask_path = dataset_folder + dataset_name + mask_folder

                save_path = 'results/predict_wholeframe_round{}_{}/{}/frame{}_{}_repeat{}_CAM/'.format(constants.round_num,
                                                                                                        constants.strategy_type,
                                                                                                        dataset_name, str(frame),
                                                                                                        model_name, str(repeat_index))
                if os.path.isdir(save_path) == 0:
                    os.makedirs(save_path)
                # ----------- CAM whole frames -----------------
                # prediction_data_generator = PredictDataGenerator(img_path, mask_path, constants.strategy_type + '_no_preprocessing', img_format=constants.img_format)
                # CAM_images, CAM_image_filenames, image_cols, image_rows, orig_cols, orig_rows = prediction_data_generator.get_expanded_whole_frames()
                # model = build_model_predict(constants, frame, repeat_index, model_name, image_rows, image_cols, orig_rows, orig_cols)
                # visualize_feature_activation_map(model, CAM_images, CAM_image_filenames, save_path)

                # ----------- CAM cropped images ----------------
                args = constants.get_args()  # get hyper parameters
                orig_images, images, masks, image_filenames = get_data_generator_MTL(
                    [dataset_name], repeat_index, args.crop_mode, constants.img_format, 'predict')

                image_rows, image_cols = images.shape[2:]
                orig_rows, orig_cols = 0, 0
                model = build_model_predict(constants, frame, repeat_index, model_name, image_rows, image_cols, orig_rows, orig_cols)

                class_list_output = model.predict(images, batch_size=1, verbose=1)
                print(class_list_output.shape, len(image_filenames))

                # thresholding prediction to calculate evaluation statistics
                class_list_output[class_list_output < 0.5] = 0
                class_list_output[class_list_output > 0] = 1

                y_pred = class_list_output[:, 0].tolist()
                mask_class_list = convert_masks_to_classes(image_rows, image_cols, masks)
                y_true = mask_class_list
                print(y_true.shape, len(y_pred))

                # Filtering - save images if it is not True Negative
                filtered_image_filenames = []
                filtered_image_index_list = []
                for i in range(len(y_true)): # len(y_true)
                    if y_true[i] == 1 and y_pred[i] == 0:
                        prefix = 'FN'
                    elif y_true[i] == 1 and y_pred[i] == 1:
                        prefix = 'TP'
                    elif y_true[i] == 0 and y_pred[i] == 1:
                        prefix = 'FP'
                    elif y_true[i] == 0 and y_pred[i] == 0:
                        prefix = 'TN'

                    # if prefix != 'TN':
                    filtered_image_filenames.append( f'{prefix}_' + image_filenames[i].split('/')[-1] )
                    filtered_image_index_list.append(i)

                images = images[filtered_image_index_list]
                masks = masks[filtered_image_index_list]
                orig_images = orig_images[filtered_image_index_list]
                print(images.shape, orig_images.shape, masks.shape)
                
                for i, image_name in enumerate(tqdm(filtered_image_filenames)):
                    visualize_feature_activation_map(model, images[i], orig_images[i], masks[i], image_name, save_path)