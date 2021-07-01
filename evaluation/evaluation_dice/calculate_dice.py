"""
Created by: Junbong Jang
Created on 9/30/2020

Unlike calculate_dice_coeff for Self-Training, this is a more generic evaluation code for comparing two model predictions
"""

import sys
sys.path.append('../../')
from UserParams import UserParams

from scipy.io import savemat
import numpy as np
import os
import cv2
import glob
import shutil
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')


smooth = 0.0001
def dice_coef(y_true, y_pred):
    # calculate the dice coefficient between ground truth and prediction
    y_true_f = K.flatten(K.round(y_true))
    y_pred_f = K.flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def extract_edge_region(image_gt, pred_image, default_size):

    edge_gt = cv2.Canny(image_gt, 100, 200)
    kernel = np.ones((default_size, default_size), np.uint8)
    dilated_edge_gt = cv2.dilate(edge_gt, kernel, iterations=1)
    # divide by 255 to shift image intensity to range [0, 1]
    image_gt = cv2.bitwise_and(image_gt / 255, dilated_edge_gt / 255)
    pred_image = cv2.bitwise_and(pred_image / 255, dilated_edge_gt / 255)
    return image_gt, pred_image

    '''
    # original method
    edge_gt = cv2.Canny(image_gt, 100, 200)
    kernel = np.ones((default_size, default_size), np.uint8)
    dilated_edge_gt = cv2.dilate(edge_gt, kernel, iterations=1)
    # divide by 255 to shift image intensity to range [0, 1]
    image_gt = cv2.bitwise_and(image_gt / 255, dilated_edge_gt / 255)
    pred_image = cv2.bitwise_and(pred_image / 255, dilated_edge_gt / 255)
    return image_gt, pred_image
    '''


def crop_border_image(img):
    # thresholding
    THRESHOLD = 127
    img[img >= THRESHOLD] = 255
    img[img < THRESHOLD] = 0

    # remove top, left borders because prediction is hazy there
    cropped_img = img[30:, 30:]
    return cropped_img


if __name__ == '__main__':
    constants = UserParams('predict')
    root_path = f'./dice_generated/round{constants.round_num}_{constants.strategy_type}'
    # if os.path.exists(root_path) and os.path.isdir(root_path):
    #     shutil.rmtree(root_path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    for repeat_index in range(constants.REPEAT_MAX):

        # TODO mark the last column index with -1 since 50 is bigger than total mask images
        dice_value = np.zeros((len(constants.model_names), 1))
        image_list_dict = {}

        for model_index in range(len(constants.model_names)):
            # model_index and dataset_index are the same for evaluation step
            ground_truth_path = f'../../assets/{constants.dataset_folders[model_index]}{constants.dataset_names[model_index]}{constants.mask_folders[model_index]}'
            for a_frame in constants.frame_list:

                suffix = f'frame{str(a_frame)}_{constants.model_names[model_index]}_repeat{str(repeat_index)}'
                # saved_path = f'{root_path}/{suffix}/'
                predict_path = f'../../models/results/predict_wholeframe_round{constants.round_num}_{constants.strategy_type}/{constants.dataset_names[model_index]}/{suffix}/'

                if os.path.isdir(predict_path):
                    mask_filenames = glob.glob(ground_truth_path + '*.png')
                    mask_filenames.sort()
                    image_list = []

                    # calculate the dice coefficient
                    for img_index in range(len(mask_filenames)):
                        filename = mask_filenames[img_index]
                        filename = filename[len(ground_truth_path):]
                        mask_gt = cv2.imread(ground_truth_path + filename, 0)
                        mask_gt = crop_border_image(mask_gt)
                        filename = filename.replace('mask', 'img')
                        image_list.append(filename)

                        predict_image = cv2.imread(predict_path + filename, 0)
                        cropped_predict_image = crop_border_image(predict_image)

                        # edge_gt, edge_pred = extract_edge_region(mask_gt, cropped_predict_image, 10)
                        # cv2.imwrite('generated/' + constants.dataset_names[model_index] + "_gt_{}.png".format(img_index), mask_gt)
                        # cv2.imwrite('generated/' + constants.dataset_names[model_index] + "_cropped_predict_image_{}.png".format(img_index), cropped_predict_image)
                        # cv2.imwrite('generated/' + constants.dataset_names[model_index] + "_gt_edge_{}.png".format(img_index), edge_gt*255)
                        # cv2.imwrite('generated/' + constants.dataset_names[model_index] + "_edge_{}.png".format(img_index), edge_pred*255)
                        # exit()

                        dice_value[model_index, img_index] = K.get_value(dice_coef(mask_gt/255, cropped_predict_image/255))
                        K.clear_session()

                    # saved the dice coefficient into the created folder.
                    image_list_dict[str(model_index)] = image_list

                else:  # skip it in this case
                    print('predict folder not found', predict_path)

        print(np.round(dice_value[:,0],4), np.mean(dice_value[:,0]))
        np.save(f'{root_path}/dice_value.npy', dice_value)
        savemat(f'{root_path}/image_list.mat', image_list_dict)


