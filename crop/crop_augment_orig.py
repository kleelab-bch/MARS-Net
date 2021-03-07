import gc
import argparse
import numpy as np
import time
from train_val_generate_orig import data_generate, augment_data, preprocess_input, preprocess_output
import os.path
from keras import backend as K
import sys
sys.path.append('..')
import constants

dataset = constants.dataset
def get_args(): 
    parser = argparse.ArgumentParser() 

    parser.add_argument("--input_size", type = int, default = 128)
    parser.add_argument("--output_size", type = int, default = 68)
    parser.add_argument("--randomseed", type = int, default = 0)
    parser.add_argument("--augmentation_factor", type = int, default = 50)


    if constants.teacher_student == 'teacher':
	    parser.add_argument("--dataset_folder", type = str, default = '../assets/')
	    parser.add_argument("--img_folder", type = str, default = '/img/')
	    parser.add_argument("--mask_folder", type = str, default = '/mask/')
	    parser.add_argument("--saved_folder", type = str, default = './crop_teacher/')
    else:  # for self training
        '''
	    parser.add_argument("--dataset_folder", type = str, default = '')
	    parser.add_argument("--img_folder", type = str, default = '../assets/' + constants.dataset[0] + '/img/')
	    parser.add_argument("--mask_folder", type = str, default = '../vUnet/average_hist/predict_wholeframe_teacher/' + constants.dataset[0] + '/processed_2_0_' + constants.dataset[0] + '/')
	    parser.add_argument("--saved_folder", type = str, default = './crop_student/')
        '''
        
        parser.add_argument("--dataset_folder", type = str, default = '')
        parser.add_argument("--img_folder", type = str, default = '../assets/' + constants.dataset[0] + '/img_unlabelled/')
        parser.add_argument("--mask_folder", type = str, default = '../vUnet/average_hist/predict_wholeframe_teacher_strat{}/'.format(constant.strategy_num) + constants.dataset[0] + '/processed_2_0_' + constants.dataset[0] + '/')
        parser.add_argument("--saved_folder", type = str, default = './crop_student_unlabelled/')
     
    parser.add_argument("--img_format", type = str, default = '.png')
    parser.add_argument("--crop_patches", type = int, default = 200)

    args = parser.parse_args()
    return args

def Training_dataset(num, iter, index, saved_folder, input_size, output_size, randomseed, augmentation_factor, img_folder, mask_folder, dataset_folder, img_format, cropped_patches):
    print(dataset[index], iter, num)
    train = data_generate(dataset[index], iter, input_size, output_size, randomseed, saved_folder, img_format, cropped_patches, dataset_folder, img_folder, mask_folder)
    imgs_train, msks_train, edgs_train, mean_value, std_value, imgsTrain_index, imgsVal_index, imgsTest_index = train.crop()
    np.save(saved_folder + dataset[index] + '_' + str(iter) + '_train_frames.npy', imgsTrain_index)
    np.savez(saved_folder + dataset[index] + '_' + str(iter) + '_std_mean.npz', mean_value, std_value)

    print(imgs_train.shape)
    train, mask, edge = augment_data(imgs_train, msks_train, edgs_train, augmentation_factor)
    print(train.shape)
    print('Data Preprocess...')
    train = preprocess_input(train, input_size, input_size, mean_value, std_value)
    mask = preprocess_output(mask, output_size, output_size)
    edge = preprocess_output(edge, output_size, output_size)
    print(time.time())
    print('Saving...')
    np.savez(saved_folder + dataset[index] + '_' + str(iter) + '_train_mask.npz', train, mask, edge)
        
    K.clear_session()
    return 

if __name__ == "__main__":
    #Defaults parameters
    args = get_args()
    if not os.path.exists(args.saved_folder):
        os.makedirs(args.saved_folder)
    for index in range(0, len(dataset),1):
        for num in range(1):
            #Training frame
            for frame in constants.frame_list:
                Training_dataset(num, frame, index, args.saved_folder, args.input_size, args.output_size, args.randomseed, args.augmentation_factor, args.img_folder, args.mask_folder, args.dataset_folder, args.img_format, args.crop_patches)
            gc.collect()
