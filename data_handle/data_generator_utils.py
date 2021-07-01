'''
Author Junbong Jang
Date 5/21/2021

utility functions for train and predict data generators
'''

import cv2
import numpy as np
import re
from joblib import Parallel, delayed


def threshold_mask_area_list(image_height, image_width, mask_area_list, threshold):
    min_mask_area_threshold = image_height * image_width * threshold
    return [mask_area > min_mask_area_threshold for mask_area in mask_area_list]


def convert_masks_to_areas(mask_list):
    mask_area_list = np.zeros(mask_list.shape[0])
    for i, mask in enumerate(mask_list):
        mask_area_list[i] = np.sum(mask > 0)
    return mask_area_list


def convert_masks_to_classes(image_height, image_width, mask_list):
    min_mask_area_threshold = image_height * image_width * 0.01
    mask_class_list = np.zeros(mask_list.shape[0])
    for i, mask in enumerate(mask_list):
        mask_class_list[i] = np.sum(mask > 0) > min_mask_area_threshold
    return mask_class_list


def read_images(image_path_list):
    # https://stackoverflow.com/questions/33778155/python-parallelized-image-reading-and-preprocessing-using-multiprocessing
    images = Parallel(n_jobs=4, verbose=1)(
        delayed(cv2.imread)(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_path_list
    )
    return images


def read_color_images(image_path_list):
    images = Parallel(n_jobs=4, verbose=1)(
        delayed(cv2.imread)(image_path, cv2.IMREAD_COLOR) for image_path in image_path_list
    )
    return images


def unison_shuffle_lists(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a = [a[i] for i in p]
    b = [b[i] for i in p]
    return a, b


def unison_shuffle_ndarrays(a, b):
    assert len(a) == len(b)

    shuffler = np.random.permutation(len(a))
    a_shuffled = a[shuffler]
    b_shuffled = b[shuffler]

    return a_shuffled, b_shuffled


# def unison_shuffle_multiple_ndarrays(*args):
    # assert len(args[0]) == len(args[0])
    # assert len(args[0]) == len(args[-1])
    #
    # shuffler = np.random.permutation(len(args[0]))
    # shuffled_args = []
    # for i in range(len(args)):
    #     shuffled_args.append(args[i][shuffler])

def unison_shuffle_multiple_ndarrays(a,b,c,d):
    assert len(a) == len(b)
    assert len(a) == len(c)
    
    shuffler = np.random.permutation(len(a))
    a_shuffled = a[shuffler]
    b_shuffled = b[shuffler]
    c_shuffled = c[shuffler]
    d_shuffled = d[shuffler]

    return a_shuffled, b_shuffled, c_shuffled, d_shuffled


def regex_find_crop_id(filename):
    regex = re.compile(r'_c\d+_')
    crop_id = regex.findall(filename)[0]  # example: '/_c40'

    return crop_id


def regex_find_frame_id(filename):
    regex = re.compile(r'/f\d+_c')
    frame_id = regex.findall(filename)[0]  # example: '/f040_c'

    return frame_id


def regex_find_prev_filenames(cur_filename, max_prev_frame_num):
    # For the given current frame, get n previous frames
    # cur_frame_id_string: '/f040_c', crop_id_string: 'c0'
    cur_frame_id_string = regex_find_frame_id(cur_filename)
    crop_id_string = regex_find_crop_id(cur_filename)

    cur_frame_id = int(cur_frame_id_string.replace('/f', '').replace('_c', ''))

    if cur_frame_id - max_prev_frame_num < 0:
        return None
    else:
        prev_filenames = []
        for prev_counter in range(1, max_prev_frame_num+1):
            prev_frame_id = f"/f{(cur_frame_id - prev_counter):03d}{crop_id_string}"
            prev_filenames.append(cur_filename.replace(f"{cur_frame_id_string.replace('_c', '')}{crop_id_string}", prev_frame_id))

        return prev_filenames