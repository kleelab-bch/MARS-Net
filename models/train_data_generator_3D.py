'''
Author Junbong Jang
Date 5/17/2021

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
import random
import numpy as np
import pickle
from joblib import Parallel, delayed
from data_generator_utils import *


def get_data_generators_3D(dataset_names, frame, repeat_index, crop_mode, img_format, process_type):
    # get img frames with mask only (every fifth frame)
    # sample some image frames and get their previous 4 frames
    # Put previous frames in a new dimension whose index represents T-n frame
    # load them and preprocess them together, which includes

    def sample_filenames_given_frame_id(filenames, sampled_frame_ids):
        sampled_img_filenames = []
        for img_filename in filenames:
            for sampled_frame in sampled_frame_ids:
                if sampled_frame in img_filename:
                    sampled_img_filenames.append(img_filename)
                    break
        sampled_img_filenames.sort()
        return sampled_img_filenames

    def get_cropped_filenames_frame(dataset_names, frame, max_prev_frame_num, repeat_index, crop_mode, img_format):
        all_sampled_img_filenames = []
        all_sampled_mask_filenames = []

        for dataset_name in dataset_names:
            crop_path = f'../crop/generated/crop_{crop_mode}_{dataset_name}/'
            crop_path_img = crop_path + f'img_repeat{repeat_index}/'
            crop_path_mask = crop_path + f'mask_repeat{repeat_index}/'

            img_filenames = glob.glob(crop_path_img + f'*_{crop_mode}' + img_format)
            mask_filenames = glob.glob(crop_path_mask + f'*_{crop_mode}' + img_format)
            assert len(img_filenames) > len(mask_filenames) * 3

            # count unique frames
            unique_frames = []
            for mask_filename in mask_filenames:
                prev_filenames = regex_find_prev_filenames(mask_filename, max_prev_frame_num)
                if frame_id not in unique_frames and prev_filenames is not None:  # exclude '/f001'
                    unique_frames.append(frame_id)

            # randomly sample a few frames
            sampled_frame_ids = random.sample(unique_frames, k=frame)  # example output: ['/f020', '/f010']
            sampled_img_filenames = sample_filenames_given_frame_id(img_filenames, sampled_frame_ids)
            sampled_mask_filenames = sample_filenames_given_frame_id(mask_filenames, sampled_frame_ids)
            print(dataset_name, sampled_frame_ids, len(sampled_img_filenames), len(sampled_mask_filenames))

            # combine filenames across datasets in a Cross-validation fold
            all_sampled_img_filenames = all_sampled_img_filenames + sampled_img_filenames
            all_sampled_mask_filenames = all_sampled_mask_filenames + sampled_mask_filenames

        all_sampled_img_filenames.sort()
        all_sampled_mask_filenames.sort()

        return all_sampled_img_filenames, all_sampled_mask_filenames


    def GetImageMask(x_filenames, y_filenames, max_prev_frame_num):
        '''
        :param x_filenames:
        :param y_filenames:
        :param max_prev_frame_num:
        :return: new_x_images has shape (instances: depth: height: width)
                 y_images has shape (instances: height: width)
        '''
        x_combined_filenames = []
        for x_filename in x_filenames:
            sampled_prev_img_filenames = regex_find_prev_filenames(x_filename, max_prev_frame_num)
            x_combined_filenames.append(x_filename)
            for sampled_prev_img_filename in sampled_prev_img_filenames:
                x_combined_filenames.append(sampled_prev_img_filename)
        print(len(x_combined_filenames))
        print(x_combined_filenames[0:5], x_combined_filenames[1000:1005])
        # first load all images in the filenames and create generator for augmenting them
        x_images = np.asarray(read_images(x_combined_filenames))
        y_images = np.asarray(read_images(y_filenames))

        # Depth represents current frame with index 0 and T-4 frame with index 4
        new_x_images = np.zeros(shape=(y_images.shape[0], (max_prev_frame_num+1), y_images.shape[1], y_images.shape[2]))

        for start_frame_num in range(max_prev_frame_num+1):
            new_x_images[:, start_frame_num, :, :] = x_images[start_frame_num::(max_prev_frame_num+1)]

        new_x_images = new_x_images[:, np.newaxis, :, :, :]
        y_images = y_images[:, np.newaxis, :, :]
        print('Batch x and y:', new_x_images.shape, y_images.shape)

        return new_x_images, y_images

    def calc_augmentation_factor(x_filenames, aug_batch_size):
        '''
        4/7/2021
        Limit augmentation to utilize only about 20GB RAM memory, which includes loaded model size
        Gradually reduce it as the number of frames increase
        :return: augmentation factor
        '''
        max_patches = 27200

        if len(x_filenames) >= max_patches:
            aug_factor = 0
        else:
            aug_factor = int((max_patches - len(x_filenames)) / aug_batch_size)

        print('calc_augmentation_factor', len(x_filenames), aug_factor)
        return aug_factor

    def getAugmentedImages(x_filenames, x_data, y_data):
        aug_batch_size = 64
        augmentation_factor = calc_augmentation_factor(x_filenames, aug_batch_size)

        aug_images = np.zeros((x_data.shape[0] + augmentation_factor * aug_batch_size, x_data.shape[1], x_data.shape[2], x_data.shape[3], x_data.shape[4]),
                              dtype=np.uint8)
        aug_masks = np.zeros((y_data.shape[0] + augmentation_factor * aug_batch_size, y_data.shape[1], y_data.shape[2], y_data.shape[3]),
                             dtype=np.uint8)
        print('getAugmentedImages aug', aug_images.shape, aug_images.dtype, aug_masks.shape, aug_masks.dtype)

        datagen = ImageDataGenerator(rotation_range=50.,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.1,
                                     zoom_range=0.1,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode='reflect')

        assert x_data.shape[3:] == y_data.shape[2:]  # their width and height must match for augmentation to be correct!
        for iteration in tqdm(range(augmentation_factor)):
            for prev_frame_num in range(x_data.shape[2]):
                for aug_image_batch in datagen.flow(x_data[:, :, prev_frame_num, :, :], batch_size=aug_batch_size, seed=iteration):
                    break
                aug_images[iteration * aug_batch_size:(iteration + 1) * aug_batch_size,:,prev_frame_num,:,:] = aug_image_batch

            for aug_mask_batch in datagen.flow(y_data, batch_size=aug_batch_size, seed=iteration):
                break
            aug_masks[iteration * aug_batch_size:(iteration + 1) * aug_batch_size] = aug_mask_batch

        aug_images[(iteration + 1) * aug_batch_size:] = x_data
        aug_masks[(iteration + 1) * aug_batch_size:] = y_data

        print('getAugmentedImages aug', aug_images.shape, aug_images.dtype, aug_masks.shape, aug_masks.dtype)
        return aug_images, aug_masks

    # -----------------------------------------------------
    max_prev_frame_num = 4
    # Get cropped image names
    x_filenames, y_filenames = get_cropped_filenames_frame(dataset_names, frame, max_prev_frame_num, repeat_index, crop_mode, img_format)

    # Get cropped Images
    x_data, y_data = GetImageMask(x_filenames, y_filenames, max_prev_frame_num)

    # Augmentation
    x_data, y_data = getAugmentedImages(x_filenames, x_data, y_data)

    # ---------- preprocessing ------------
    if process_type == 'normalize':
        x_data = normalize_input(x_data)
    elif process_type == 'standardize':
        x_data = preprocess_input(x_data)
    else:
        raise Exception('incorrect process_type {}'.format(process_type))

    y_data = preprocess_output(y_data)
    y_data = y_data[:, :, 30:y_data.shape[3] - 30, 30:y_data.shape[3] - 30]
    x_data, y_data = unison_shuffle_ndarrays(x_data, y_data)

    # split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, shuffle=True, test_size=0.2, random_state=repeat_index)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    # separate depth channel in x_data into current and T-1 ~ T-4 frames
    x_train_list = []
    x_val_list = []
    for frame_counter in range(max_prev_frame_num+1):
        x_train_list.append(x_train[:, :, frame_counter, :, :])
        x_val_list.append(x_val[:, :, frame_counter, :, :])

    return x_train_list, y_train, x_val_list, y_val


# ------------------------------------------------------------

def get_data_generators_3D_all(dataset_names, repeat_index, crop_mode, img_format, process_type, input_depth):
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
        def filter_filename_indices_with_crop_id(filenames, dataset_name, unique_id):
            filtered_filename_indices = []
            for index, filename in enumerate(filenames):
                if unique_id in filename and (dataset_name + '/') in filename:
                    filtered_filename_indices.append(index)
            return filtered_filename_indices

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

                # Split 3D images by predefined input depth for training
                # This procedure exclude a few frames in the middle
                for split_num in range(crop_movie_split_num):
                    if split_num == crop_movie_split_num - 1:
                        batch_x_indices = filtered_x_filename_indices[num_frames - input_depth:]
                    else:
                        batch_x_indices = filtered_x_filename_indices[
                                          input_depth * split_num:input_depth * (1 + split_num)]

                    all_data_x[data_counter] = data_x[batch_x_indices]
                    all_data_y[data_counter] = data_y[batch_x_indices]
                    data_counter = data_counter + 1

        batch_x = all_data_x[:, np.newaxis, :, :, :]
        batch_y = all_data_y[:, np.newaxis, :, :, :]
        print('Processed Input Data:', batch_x.shape, batch_y.shape)

        return batch_x, batch_y


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


