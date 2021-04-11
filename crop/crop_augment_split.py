import gc
import numpy as np
import time

from train_val_generate_split import data_generate, augment_data, heq_norm_input, normalize_input, normalize_clip_input, expand_channel_input, preprocess_input, preprocess_output
import os.path
from tensorflow.keras import backend as K
import gc
import math
import psutil
import sys
sys.path.append('..')
from UserParams import UserParams


def Training_dataset(constants, model_name, dataset_name, frame, repeat_index, input_size, output_size, batch_size, augmentation_factor, img_folder, mask_folder, dataset_folder, img_format, crop_mode, cropped_patches):
    print('Training dataset: ', dataset_name, frame)
    train = data_generate(dataset_name, frame, input_size, output_size, repeat_index, img_format, crop_mode, cropped_patches, dataset_folder, img_folder, mask_folder)
    imgs_train, masks_train, edges_train, mean_value, std_value, imgsTrain_index, imgsTest_index, framenames = train.crop()

    print(imgs_train.shape, masks_train.shape, edges_train.shape)

    img_counts, img_row, img_col = imgs_train.shape
    if crop_mode == 'even' and batch_size > img_counts:
        batch_size = img_counts
    split_constant = constants.crop_split_constant
    for split_index in range(split_constant):
        split_lower_limit = math.ceil(img_counts/split_constant*split_index)
        split_upper_limit = math.ceil(img_counts/split_constant*(split_index+1))
        print('split', split_lower_limit, split_upper_limit)
        if split_index == split_constant-1:
            split_imgs = imgs_train[split_lower_limit:,:,:]
            split_msks = masks_train[split_lower_limit:,:,:]
            split_edgs = edges_train[split_lower_limit:,:,:]
        else:
            split_imgs = imgs_train[split_lower_limit:split_upper_limit,:,:]
            split_msks = masks_train[split_lower_limit:split_upper_limit,:,:]
            split_edgs = edges_train[split_lower_limit:split_upper_limit,:,:]
        print('split_imgs shape: ', split_imgs.shape)


        train, mask, edge = augment_data(split_imgs, split_msks, split_edgs, batch_size, augmentation_factor)
        print('train shape: ', train.shape, ' mask shape: ', mask.shape)

        print('Image Preprocessing...')
        if 'no_preprocessing' in str(constants.strategy_type):
            train = expand_channel_input(train, input_size, input_size)
        elif 'normalize_clip' in str(constants.strategy_type):
            train = normalize_clip_input(train, input_size, input_size, mean_value, std_value)
        elif 'normalize' in str(constants.strategy_type):
            train = normalize_input(train, input_size, input_size)
        elif '_heq' in str(constants.strategy_type):
            train = heq_norm_input(train, input_size, input_size)
        else:
            train = preprocess_input(train, input_size, input_size, mean_value, std_value)

        print(f'mean: {np.around(np.mean(train), 3)}  std: {np.around(np.std(train), 3)}')

        mask = preprocess_output(mask, output_size, output_size)
        edge = preprocess_output(edge, output_size, output_size)

        print('Saving...')
        root_path, save_suffix = constants.get_crop_path(model_name, dataset_name, str(frame), str(split_index), str(repeat_index))
        np.savez(root_path + save_suffix, train, mask, edge)

        print('Clear memory')
        print(psutil.virtual_memory())
        print(gc.collect(), end='\n\n')  # runs garbage collection to free memory

    np.save('{}{}_frame{}_repeat{}_train_index.npy'.format(root_path, dataset_name, frame, repeat_index), imgsTrain_index)
    np.save('{}{}_frame{}_repeat{}_train_names.npy'.format(root_path, dataset_name, frame, repeat_index), framenames)
    np.savez('{}{}_frame{}_repeat{}_std_mean.npz'.format(root_path, dataset_name, frame,repeat_index), mean_value, std_value)

    K.clear_session()
    return

if __name__ == "__main__":
    constants = UserParams('crop')

    args = constants.get_crop_args()
    print(args)
    for repeat_index in range(constants.REPEAT_MAX):
        for frame in constants.frame_list:
            for model_name in constants.model_names:
                # for teacher model training, skip other models
                if model_name != constants.model_names[0] and constants.self_training_type is None:
                    print('break:', repeat_index, frame, model_name)
                    break
                for dataset_folder, dataset_name, img_folder, mask_folder in zip(constants.dataset_folders, constants.dataset_names, constants.img_folders, constants.mask_folders):
                    Training_dataset(constants, model_name, dataset_name, frame, repeat_index, args.input_size, args.output_size, args.batch_size, args.augmentation_factor, img_folder, mask_folder, dataset_folder, constants.img_format, args.crop_mode, args.crop_patches)
            gc.collect()  # runs garbage collection to free memory
