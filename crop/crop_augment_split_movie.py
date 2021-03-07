import gc
import argparse
import numpy as np
import time
from train_val_generate_split_movie import data_generate, augment_data, preprocess_input, preprocess_output
import os.path
from tensorflow.keras import backend as K
import gc
import math
import psutil
import sys
sys.path.append('..')
from UserParams import UserParams

def get_args(constants):
    parser = argparse.ArgumentParser() 

    parser.add_argument("--input_size", type = int, default = 128)
    parser.add_argument("--output_size", type = int, default = 68)
    parser.add_argument("--randomseed", type = int, default = 0)
    parser.add_argument("--img_format", type = str, default = constants.img_format)
    parser.add_argument("--crop_patches", type = int, default = 200)
    parser.add_argument("--augmentation_factor", type = int, default = 50)
    parser.add_argument("--dataset_folder", type = str, default = constants.dataset_folder)
    parser.add_argument("--img_folder", type = str, default = constants.img_folder)
    parser.add_argument("--saved_folder", type = str, default = './crop_results/crop_round{}_{}/'.format(constants.round_num, constants.strategy_type))
    
    args = parser.parse_args()
    return args

def Training_dataset(frame, dataset_index, saved_folder, input_size, output_size, randomseed, augmentation_factor, img_folder, mask_folder, dataset_folder, img_format, cropped_patches):
    constants = UserParams('crop')
    print('Training dataset: ', constants.dataset[dataset_index], frame)
    train = data_generate(constants.dataset[dataset_index], frame, input_size, output_size, randomseed, saved_folder, img_format, cropped_patches, dataset_folder, img_folder, mask_folder)
    imgs_train, msks_train, edgs_train, mean_value, std_value, imgsTrain_index, imgsVal_index, imgsTest_index = train.crop()

    np.save(saved_folder + constants.dataset[dataset_index] + '_' + str(frame) + '_train_frames.npy', imgsTrain_index)
    np.savez(saved_folder + constants.dataset[dataset_index] + '_' + str(frame) + '_std_mean.npz', mean_value, std_value)

    img_counts = imgs_train.shape[0]
    print(imgs_train.shape)
    split_constant = constants.crop_split_constant
    for split_index in range(split_constant):
        split_lower_limit = math.ceil(img_counts/split_constant*split_index)
        split_upper_limit = math.ceil(img_counts/split_constant*(split_index+1))
        print('split', split_lower_limit, split_upper_limit)
        if split_index == split_constant-1:
            split_imgs = imgs_train[split_lower_limit:]
            split_msks = msks_train[split_lower_limit:]
            split_edgs = edgs_train[split_lower_limit:]
        else:
            split_imgs = imgs_train[split_lower_limit:split_upper_limit]
            split_msks = msks_train[split_lower_limit:split_upper_limit]
            split_edgs = edgs_train[split_lower_limit:split_upper_limit]
        print('split shape: ', split_imgs.shape, split_msks.shape, split_edgs.shape)
        
        train, mask, edge = augment_data(split_imgs, split_msks, split_edgs, augmentation_factor)
        print('augmented shapes: ', train.shape, mask.shape, edge.shape)
        
        print('Image Preprocessing...')
        train = preprocess_input(train, input_size, input_size, mean_value, std_value)
        mask = preprocess_output(mask, output_size, output_size)
        edge = preprocess_output(edge, output_size, output_size)
        print('processed shapes: ', train.shape, mask.shape, edge.shape)
        
        print('Saving...')
        np.savez(saved_folder + constants.dataset[dataset_index] + '_' + str(frame) + '_split' + str(split_index) + '_train_mask.npz', train, mask, edge)
        
        print('Clear memory')
        print(psutil.virtual_memory())
        print(gc.collect(), end='\n\n')  # runs garbage collection to free memory
        
    K.clear_session()
    return 

if __name__ == "__main__":
    constants = UserParams('crop')
    #Defaults parameters
    args = get_args(constants)
    if not os.path.exists(args.saved_folder):
        os.makedirs(args.saved_folder)
    for frame in constants.frame_list:        
        for dataset_index in range(len(constants.dataset)):
            mask_folder = constants.get_mask_folder(dataset_index)
            Training_dataset(frame, dataset_index, args.saved_folder, args.input_size, args.output_size, args.randomseed, args.augmentation_factor, args.img_folder, mask_folder, args.dataset_folder, args.img_format, args.crop_patches)
        gc.collect()  # runs garbage collection to free memory
