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
    # calculate the dice coefficient between ground truth and predi
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

    '''
    edge_gt = cv2.Canny(image_gt, 100, 200)
    edge_pred = cv2.Canny(pred_image, 100, 200)
    kernel = np.ones((default_size, default_size), np.uint8)
    dilated_edge_gt = cv2.dilate(edge_gt, kernel, iterations=1)
    dilated_edge_pred = cv2.dilate(edge_pred, kernel, iterations=1)
    dilated_edge_region = cv2.bitwise_or(dilated_edge_gt, dilated_edge_pred)

    # cv2.imwrite('generated/dilated_edge_region.png'.format(img_index), dilated_edge_region)

    # divide by 255 to shift image intensity to range [0, 1]
    image_gt = cv2.bitwise_and(image_gt / 255, dilated_edge_region / 255)
    pred_image = cv2.bitwise_and(pred_image / 255, dilated_edge_region / 255)
    return image_gt, pred_image
    '''

    '''
    # canny and dillation withut bitwise_And
    default_size=3
    edge_gt = cv2.Canny(image_gt,0,0)
    round1_edge = cv2.Canny(round1_image,0,0)

    kernel = np.ones((default_size,default_size),np.uint8)
    edge_gt = cv2.dilate(edge_gt,kernel,iterations = 1)
    round1_edge = cv2.dilate(round1_edge,kernel,iterations = 1)

    edge_gt = edge_gt/255
    round1_edge = round1_edge/255
    return edge_gt, round1_edge
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
    constants = UserParams('eval')
    root_path = './dice_coeff_compare/' + constants.get_eval_folder_name() + '/'
    # if os.path.exists(root_path) and os.path.isdir(root_path):
    #     shutil.rmtree(root_path)

    for prediction_path_index, prediction_path in enumerate(constants.predict_path_list):
        constants.update_eval_config(prediction_path_index)

        for repeat_index in range(constants.REPEAT_MAX):

            for model_index in range(len(constants.model_names)):
                # model_index and dataset_index are the same for evaluation step
                ground_truth_path = constants.dataset_folder + constants.dataset_names[model_index] + constants.mask_folder
                for a_frame in constants.frame_list:
                    print()
                    print('predict_index: ', prediction_path_index, ' repeat_index: ', repeat_index, ' model_index: ', model_index, ' a_frame: ', a_frame)
                    print()

                    save_suffix = '/frame{}_{}_repeat{}/'.format(str(a_frame), constants.model_names[model_index], str(repeat_index))
                    saved_path = root_path + save_suffix

                    load_suffix = '/processed_frame{}_{}_repeat{}/'.format(str(a_frame), constants.model_names[model_index], str(repeat_index))
                    predict_folder = prediction_path + constants.dataset_names[model_index] + load_suffix

                    if os.path.isdir(predict_folder):
                        if not os.path.exists(saved_path):
                            os.makedirs(saved_path)
                        mask_filenames = glob.glob(ground_truth_path + '*.png')
                        mask_filenames.sort()

                        dice_value = np.zeros((1, len(mask_filenames)))
                        image_list = []

                        # calculate the dice coefficient
                        for img_index in range(len(mask_filenames)):
                            filename = mask_filenames[img_index]
                            filename = filename[len(ground_truth_path):]
                            image_gt = cv2.imread(ground_truth_path + filename, 0)
                            image_gt = crop_border_image(image_gt)
                            filename = filename.replace('mask', 'img')
                            image_list.append(filename)

                            predict_image = cv2.imread(predict_folder + filename, 0)
                            cropped_predict_image = crop_border_image(predict_image)

                            # edge_gt, edge_pred = extract_edge_region(image_gt, cropped_predict_image, 10)
                            # cv2.imwrite('generated/' + constants.dataset_names[model_index] + "_gt_{}.png".format(img_index), image_gt)
                            # cv2.imwrite('generated/' + constants.dataset_names[model_index] + "_cropped_predict_image_{}.png".format(img_index), cropped_predict_image)
                            # cv2.imwrite('generated/' + constants.dataset_names[model_index] + "_gt_edge_{}.png".format(img_index), edge_gt*255)
                            # cv2.imwrite('generated/' + constants.dataset_names[model_index] + "_edge_{}.png".format(img_index), edge_pred*255)
                            # exit()

                            # dice_value[0, img_index] = K.get_value(dice_coef(edge_gt, edge_pred))
                            dice_value[0, img_index] = K.get_value(dice_coef(image_gt/255, cropped_predict_image/255))
                            K.clear_session()

                        # saved the dice coefficient into the created folder.
                        np.save(saved_path + f'model{prediction_path_index}_dice_value.npy', dice_value)
                        mdic = {"image_list": image_list}
                        savemat(saved_path + f'model{prediction_path_index}_image_list.mat', mdic)

                    else:  # skip it in this case
                        print('predict folder not found', predict_folder)


