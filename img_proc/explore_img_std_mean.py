'''
Author junbong Jang
8/10/2020

To see the pixel intensity distribution of images in a dataset folder
'''


import glob
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # For running in putty without showing the figure, Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from UserParams import UserParams


def get_img_list(dataset_path):
    img_path_list = glob.glob(dataset_path+'*.png')
    img = cv2.imread(img_path_list[0], cv2.IMREAD_GRAYSCALE)
    img_r, img_c = img.shape
    total_number = len(img_path_list)
    
    img_list = np.ndarray((total_number, img_r, img_c), dtype=np.uint8)
    for i in range(total_number):
        img_name = img_path_list[i]
        img_list[i] = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    
    return img_list

    
def draw_img_histogram(img, title, save_path):
    avg = np.mean(np.ravel(img))
    std = np.std(np.ravel(img))
    text_input = 'avg: %.3f' % np.around(avg,3) + '\nsd: %.3f' % np.around(std,3)
    print('draw_img_histogram: ')
    print(text_input)

    fig, ax = plt.subplots()
    plt.hist(img, bins=100)
    plt.text(x=0.02, y=.9, s=text_input, color='black',
            bbox=dict(facecolor='none', edgecolor='black', pad=5.0), transform=ax.transAxes)
    
    plt.title(title, fontsize='large')
    plt.xlabel('Standardized Pixel Intensity', fontsize='large')
    plt.ylabel('Frequency', fontsize='large')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    
def get_custom_avg_std(img):
    avg_list = []
    sd_list = []
    for img in img_list:
        avg_list.append(np.mean(img))
        sd_list.append(np.var(img))
    print('average:',  round(np.mean(avg_list),3), '   standard dev:', round(np.sqrt(np.mean(sd_list)),3))    


def normalize_clip_img_list(img_list, input_type):
    avg = np.mean(img_list)
    std = np.std(img_list)
    print(a_dataset, f'average: {np.around(avg,3)}    standar dev: {np.around(std,3)}')

    max_val = avg + 3 * std
    min_val = avg - 3 * std
    if input_type != 'z-score':
        if min_val < 0:
            min_val = 0
        if max_val > 255:
            max_val = 255
        print('min, max:', min_val, max_val)
        np.clip(img_list, min_val, max_val, out=img_list)

    normalized_img_list = (img_list - min_val) / (max_val - min_val)

    return normalized_img_list


def normalize_img_list(img_list):
    avg = np.mean(img_list)
    std = np.std(img_list)
    print(a_dataset, f'average: {np.around(avg,3)}    standar dev: {np.around(std,3)}')

    normalized_img_list = img_list / 255

    return normalized_img_list


def standardize_img_list(img_list):
    avg = np.mean(img_list)
    std = np.std(img_list)
    print(a_dataset, f'average: {np.around(avg,3)}    standar dev: {np.around(std,3)}')

    # max_val = avg + 3 * std
    # min_val = avg - 3 * std
    # if min_val < 0:
    #     min_val = 0
    # if max_val > 255:
    #     max_val = 255
    # print('min, max:', min_val, max_val)
    # np.clip(img_list, min_val, max_val, out=img_list)
    # avg = np.mean(img_list)
    # std = np.std(img_list)

    standardized_img_list = (img_list - avg) / std

    return standardized_img_list


if __name__ == "__main__":
    # Compare how image preprocessing affects image pixel intensity distribution
    constants = UserParams('crop')

    for a_dataset in constants.dataset_names:
        dataset_path = constants.dataset_folder + a_dataset + constants.img_folder
        img_list = get_img_list(dataset_path)

        # Standardize the test set images with test set's avg and std
        # standardized_img_list = standardize_img_list(img_list)
        # draw_img_histogram(np.ravel(standardized_img_list), 'Z-Score Norm: ' + a_dataset, 'generated/z_score_norm_' + a_dataset + '.png')

        # Min Max Normalize the test set images
        normalized_img_list = normalize_img_list(img_list)
        draw_img_histogram(np.ravel(normalized_img_list), 'Min Max Norm: ' + a_dataset, 'generated/min_max_norm_' + a_dataset + '.png')

        # Min Max Normalize Clip the test set images
        # normalized_img_list = normalize_clip_img_list(img_list, input_type='original')
        # draw_img_histogram(np.ravel(normalized_img_list), 'Min Max Norm: ' + a_dataset, 'generated/min_max_norm_clip_' + a_dataset + '.png')

        # standardize first and then normalize
        # standardized_img_list = standardize_img_list(img_list)
        # normalized_img_list = normalize_img_list(standardized_img_list, input_type='z-score')
        # print(np.unique(normalized_img_list))
        # draw_img_histogram(np.ravel(normalized_img_list), 'Z-Score And Min Max Norm: ' + a_dataset, 'generated/z_score_min_max_norm' + a_dataset + '.png')

        # histogram equalization and then normalize



    # Normalize the test set images with training set's avg and std
    # for model_index, model_name in enumerate(constants.model_names):
    #     model_data_list = np.array([])
    #     print(model_name, 'model_index: ', model_index)
    #     for dataset_index, a_dataset in enumerate(constants.dataset):
    #         if dataset_index != model_index:
    #             dataset_path = constants.dataset_folder + a_dataset + constants.img_folder
    #             img_list = get_img_list(dataset_path)
    #             print(a_dataset, 'dataset_index: ', dataset_index)
    #             if dataset_index == 0:
    #                 model_data_list = np.ravel(img_list)
    #             else:
    #                 model_data_list = np.concatenate((model_data_list, np.ravel(img_list)))
    #
    #     avg = np.mean(np.ravel(model_data_list))
    #     std = np.std(np.ravel(model_data_list))
    #     print('model datasets\' average:',  round(avg,3), '   standard dev:', round(std,3))
    #
    #     dataset_path = constants.dataset_folder + a_dataset + constants.img_folder
    #     test_img_list = get_img_list(dataset_path)
    #     standardized_img_list = (test_img_list - avg) / std
    #
    #     draw_img_histogram(np.ravel(standardized_img_list), 'Mean, Sd: ' + model_name + '\nStandardize: ' + constants.dataset[model_index], 'generated/standardized_' + model_name + '.png')


