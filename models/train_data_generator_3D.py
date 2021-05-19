'''
Author Junbong Jang
Date 3/14/2021

preprocess cropped images by shuffling, splitting (training, validation, and test sets) and standardizing
augment preprocessed images in-memory
'''
from data_processor import preprocess_input, preprocess_output, normalize_input, preprocess_per_input_image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')

from tqdm import tqdm
import glob
import re
import random
import numpy as np
import pickle
import cv2
from joblib import Parallel, delayed


def get_data_generators_3D(dataset_names, repeat_index, crop_mode, img_format, process_type, input_depth):
    # retrieve cropped images
    x_filenames, y_filenames, unique_crop_id_per_dataset = get_cropped_filenames(dataset_names, repeat_index, crop_mode, img_format)
    x_data, y_data = GetImageMask(x_filenames, y_filenames, unique_crop_id_per_dataset, input_depth)

    # ---------- preprocessing ------------
    if process_type == 'normalize':
        x_data = normalize_input(x_data)
    elif process_type == 'standardize':
        x_data = preprocess_input(x_data)
    else:
        raise Exception('incorrect process_type {}'.format(process_type))

    y_data = preprocess_output(y_data)
    y_data = y_data[:, :, :, 30:y_data.shape[3] - 30, 30:y_data.shape[3] - 30]
    x_data, y_data = unison_shuffle_ndarrays(x_data, y_data)

    # split data
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, shuffle=True, test_size=0.2, random_state=repeat_index)

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    return x_train, y_train, x_val, y_val


def get_cropped_filenames(dataset_names, repeat_index, crop_mode, img_format):
    all_img_filenames = []
    all_mask_filenames = []
    unique_crop_id_per_dataset = {}

    for dataset_name in dataset_names:
        crop_path = f'../crop/generated/crop_{crop_mode}_{dataset_name}/'
        crop_path_img = crop_path + f'img_repeat{repeat_index}/'
        crop_path_mask = crop_path + f'mask_repeat{repeat_index}/'

        img_filenames = glob.glob(crop_path_img + f'*_{crop_mode}' + img_format)
        mask_filenames = glob.glob(crop_path_mask + f'*_{crop_mode}' + img_format)
        assert len(img_filenames) == len(mask_filenames)

        # count unique crops
        unique_crop_id_list = []
        for img_filename in img_filenames:
            crop_id = regex_find_crop_id(img_filename)
            if crop_id not in unique_crop_id_list:
                unique_crop_id_list.append(crop_id)
        unique_crop_id_per_dataset[dataset_name] = unique_crop_id_list

        # sampling img and mask filenames
        # filenames for image and mask are the same
        all_img_filenames = all_img_filenames + img_filenames
        all_mask_filenames = all_mask_filenames + mask_filenames
    all_img_filenames.sort()
    all_mask_filenames.sort()

    return all_img_filenames, all_mask_filenames, unique_crop_id_per_dataset


def GetImageMask(x_filenames, y_filenames, unique_crop_id_per_dataset, input_depth):
    # pre-compute how many cropped movies there will be to preallocate numpy array
    total_crop_num = 0
    for dataset_name in unique_crop_id_per_dataset.keys():
        for crop_id in unique_crop_id_per_dataset[dataset_name]:
            filtered_x_filename_indices = filter_filename_indices_with_crop_id(x_filenames, dataset_name, crop_id)
            filtered_y_filename_indices = filter_filename_indices_with_crop_id(y_filenames, dataset_name, crop_id)
            assert filtered_x_filename_indices == filtered_y_filename_indices

            num_frames = len(filtered_x_filename_indices)
            crop_movie_split_num = round(num_frames / input_depth)  # probably 4
            total_crop_num = total_crop_num + crop_movie_split_num
    print('total_crop_num', total_crop_num)

    # load all images in the filenames
    data_x = np.asarray(read_images(x_filenames))
    data_y = np.asarray(read_images(y_filenames))
    all_data_x = np.zeros((total_crop_num, input_depth, data_x.shape[2], data_x.shape[2]), dtype=np.uint8)
    all_data_y = np.zeros((total_crop_num, input_depth, data_y.shape[2], data_y.shape[2]), dtype=np.uint8)

    data_counter = 0
    for dataset_name in unique_crop_id_per_dataset.keys():
        for crop_id in tqdm(unique_crop_id_per_dataset[dataset_name]):
            filtered_x_filename_indices = filter_filename_indices_with_crop_id(x_filenames, dataset_name, crop_id)
            num_frames = len(filtered_x_filename_indices)
            crop_movie_split_num = round(num_frames / input_depth)  # probably 4

            for split_num in range(crop_movie_split_num):
                if split_num == crop_movie_split_num-1:
                    batch_x_indices = filtered_x_filename_indices[num_frames - input_depth:]
                else:
                    batch_x_indices = filtered_x_filename_indices[input_depth*split_num:input_depth*(1+split_num)]

                all_data_x[data_counter] = data_x[batch_x_indices]
                all_data_y[data_counter] = data_y[batch_x_indices]
                data_counter = data_counter + 1

    batch_x = all_data_x[:,np.newaxis,:,:,:]
    batch_y = all_data_y[:,np.newaxis,:,:,:]
    print('Processed Input Data:', batch_x.shape, batch_y.shape)

    return batch_x, batch_y


def filter_filename_indices_with_crop_id(filenames, dataset_name, unique_id):
    filtered_filename_indices = []
    for index, filename in enumerate(filenames):
        if unique_id in filename and (dataset_name + '/') in filename:
            filtered_filename_indices.append(index)
    return filtered_filename_indices


def read_images(image_path_list):
    # https://stackoverflow.com/questions/33778155/python-parallelized-image-reading-and-preprocessing-using-multiprocessing
    images = Parallel(n_jobs=4, verbose=1)(
        delayed(cv2.imread)(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_path_list
    )
    return images


def unison_shuffle_ndarrays(a, b):
    assert len(a) == len(b)

    shuffler = np.random.permutation(len(a))
    a_shuffled = a[shuffler]
    b_shuffled = b[shuffler]

    return a_shuffled, b_shuffled


def regex_find_crop_id(filename):
    regex = re.compile(r'_c\d+_')
    crop_id = regex.findall(filename)[0]  # example: '/_c40'

    return crop_id