'''
Author Junbong Jang
Creation Date 3/9/2021

processing functions for images and masks before training, or prediction
'''

import numpy as np
import cv2
import statistics
import math
import glob

import sys
sys.path.append('..')


def to3channel(imgs):
    if imgs.shape[1] == 1:
        imgs = np.repeat(imgs, 3, axis=1)

    return imgs.astype('float32')


def preprocess_output(imgs):
    imgs_p = imgs.astype('float32')
    imgs_p /= 255.  # scale masks to [0, 1]

    return imgs_p


def preprocess_per_input_image(imgs):
    print('preprocess_per_input_image')
    imgs = to3channel(imgs)
    for img_index in range(imgs.shape[0]):
        img = imgs[img_index]
        std = np.std(img)
        mean = np.mean(img)

        img -= mean
        img /= std
        imgs[img_index] = img
    return imgs


def preprocess_input(imgs, std=None, mean=None):
    imgs_p = to3channel(imgs)
    if std is None:
        std = np.std(imgs_p)
    if mean is None:
        mean = np.mean(imgs_p)
    print('preprocess_input', std, mean)
    imgs_p -= mean
    imgs_p /= std

    return imgs_p


def minmax_normalize_per_input_image(imgs):
    imgs_p = to3channel(imgs)
    for img_index in range(imgs_p.shape[0]):  # scale each image to [0, 1]
        img = imgs_p[img_index]
        max = np.amax(img)
        min = np.amin(img)
        if max == min:
            print(f'minmax_normalize_per_input_image max == min, index {img_index}')
            # raise Exception(f'image index {img_index}\n, max == min, {max} {min}', img)
            img = (img - min) / min
        else:
            img = (img - min) / (max - min)
        imgs_p[img_index] = img

    return imgs_p


def normalize_input(imgs):
    imgs_p = to3channel(imgs)
    imgs_p /= 255.  # scale image to [0, 1]
    return imgs_p


def clip_input(imgs):
    imgs_p = to3channel(imgs)
    std, mean = np.std(imgs_p), np.mean(imgs_p)

    min_val = mean - 2 * std
    max_val = mean + 3 * std
    if min_val < 0:
        min_val = 0
    if max_val > 255:
        max_val = 255
    np.clip(imgs_p, min_val, max_val, out=imgs_p)
    # imgs_p = (imgs_p - min_val) / (max_val - min_val)

    return imgs_p


def heq_norm_input(imgs):
    imgs_heq = np.ndarray(imgs.shape, dtype=np.uint8)
    for img_index in range(imgs_heq.shape[0]):
        imgs_heq[img_index,0] = cv2.equalizeHist(imgs[img_index,0])

    imgs_p = to3channel(imgs_heq)
    imgs_p /= 255.  # scale image to [0, 1]

    return imgs_p

# -----------------------------------------
# ------ about std_mean_from_images -------
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
    for dataset_index in range(0, len(constants.dataset), 1):
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
    frame_std_value = math.sqrt(statistics.mean(square(frame_std_list)))
    return frame_std_value, frame_mean_value


def aggregate_std_mean(dataset_names, excluded_dataset_name, frame, repeat_index, crop_path):
    # for self training five fold validation,
    # get average of std and mean from four movies to preprocess the test set images.
    print('aggregate_std_mean')
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
            print(dataset_names[dataset_index], std_value, mean_value)
    frame_mean_value = statistics.mean(frame_mean_list)
    frame_std_value = math.sqrt(statistics.mean(square(frame_std_list)))
    print(frame_std_value, frame_mean_value)
    return frame_std_value, frame_mean_value


def get_std_mean_from_train_images(dataset_folder, dataset_names, img_folder, img_format):
    '''
    7/28/2021
    specifically written for spheroid_test_VGG19_marsnet prediction
    :param dataset_folder:
    :param dataset_names:
    :param img_format:
    :return:
    '''
    print('get_std_mean_from_train_images')
    all_img_list = []
    for i, dataset_name in enumerate(dataset_names):
        train_img_path = dataset_folder + '/' + dataset_name + img_folder
        img_list = glob.glob(train_img_path + '*' + img_format)

        if len(img_list) == 0:  # skip this dataset
            print('img list is empty')
            exit()

        # img = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
        # img_r, img_c = img.shape
        # total_number = len(img_list)
        # imgs = np.ndarray((total_number, img_r, img_c), dtype=np.uint8)
        for img_index in range(len(img_list)):
            img_path = img_list[img_index]
            img_name = img_path[len(train_img_path):]
            # imgs[img_index] = cv2.imread(train_img_path + img_name, cv2.IMREAD_GRAYSCALE)
            all_img_list.append(cv2.imread(train_img_path + img_name, cv2.IMREAD_GRAYSCALE))
    print(len(all_img_list))
    imgs = np.stack(all_img_list, axis=0)

    avg = np.mean(imgs)
    std = np.std(imgs)
    print(imgs.shape, avg, std)
    return std, avg


def get_std_mean_from_images(all_img_path, img_format):
    img_list = glob.glob(all_img_path + '*' + img_format)

    if len(img_list) == 0:  # skip this dataset
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
