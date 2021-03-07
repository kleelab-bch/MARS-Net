'''
Junbong Jang
5/16/2020

To see how std and mean differs per dataset
'''

import numpy as np
import statistics 
import math
import glob
import cv2

import sys
sys.path.append('..')

def square(list):
    return [i ** 2 for i in list]


def get_rest_indices_from_all(all_indices, chosen_index):
    '''
    given a list of indices, and one chosen dataset index,
    get indices other than the chosen dataset index
    '''
    rest_indices = set(all_indices) - set([chosen_index])
    return list(rest_indices)


def loop_aggregate_std_mean(constants):
    for dataset_index in range(0, len(constants.dataset),1):
        for frame in constants.frame_list:
            aggregate_std_mean(constants, dataset_index, frame)
            

def aggregate_std_mean_except(constants, dataset_index, frame, crop_path):
    print(constants.model_names[dataset_index], end=' ')
    print(frame)
    frame_mean_list = []
    frame_std_list = []
    rest_indices = get_rest_indices_from_all(range(len(constants.dataset)), dataset_index)
    for rest_index in rest_indices:
        std_mean = np.load(crop_path + constants.dataset[rest_index] + '_' + str(frame) + '_std_mean.npz')
        mean_value = std_mean['arr_0'].tolist()
        std_value = std_mean['arr_1'].tolist()
        
        frame_mean_list.append(mean_value)
        frame_std_list.append(std_value)
        print(constants.dataset[rest_index], mean_value, std_value)
    frame_mean_value = statistics.mean(frame_mean_list)
    frame_std_value =  math.sqrt(statistics.mean(square(frame_std_list)))
    return frame_std_value, frame_mean_value
    

def aggregate_std_mean(dataset_names, excluded_dataset_name, frame, repeat_index, crop_path):
    # for self training five fold validation, 
    # get average of std and mean from four movies to preprocess the test set images.
    print('aggregate_std_mean:' + str(frame))
    frame_mean_list = []
    frame_std_list = []

    for dataset_index in range(len(dataset_names)):
        if dataset_names[dataset_index] != excluded_dataset_name:
            save_suffix = '{}_frame{}_repeat{}'.format(dataset_names[dataset_index], str(frame), str(repeat_index))
            std_mean = np.load(crop_path + save_suffix + '_std_mean.npz')
            mean_value = std_mean['arr_0'].tolist()
            std_value = std_mean['arr_1'].tolist()

            frame_mean_list.append(mean_value)
            frame_std_list.append(std_value)
            print(dataset_names[dataset_index], mean_value, std_value)
    frame_mean_value = statistics.mean(frame_mean_list)
    frame_std_value =  math.sqrt(statistics.mean(square(frame_std_list)))
    return frame_std_value, frame_mean_value
            
            
def get_std_mean_from_images(all_img_path, img_format):
    img_list = glob.glob(all_img_path + '*' + img_format)
    
    if len(img_list) == 0: # skip this dataset
        print('img list is empty')
        exit()
    
    img = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
    img_r, img_c = img.shape
    total_number = len(img_list)
    imgs = np.ndarray((total_number, img_r, img_c), dtype=np.uint8)
    for i in range(len(img_list)):
        img_path = img_list[i]
        img_name = img_path[len(all_img_path):]
        imgs[i] = cv2.imread(all_img_path + img_name, cv2.IMREAD_GRAYSCALE)

    avg = np.mean(imgs)
    std = np.std(imgs)
    return std, avg 


if __name__ == "__main__":
    #loop_aggregate_std_mean()
    print_all_std_mean()
