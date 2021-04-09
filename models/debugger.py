'''
Author Junbong Jang
Date 9/2/2020

Contains debugging tools for deep learning research

'''
import sys
sys.path.append('..')
from UserParams import UserParams
from data_processor import get_std_mean_from_images

import os
import cv2
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K


def check_loaded_weights(weights_path):
    loaded_weights = h5py.File(weights_path)
    allKeys = list(loaded_weights.keys())

    print('check_loaded_weights')
    print(loaded_weights[allKeys[0]].name)
    for hi in loaded_weights[allKeys[0]]:
        print(hi)

    print()
    for hi in loaded_weights[allKeys[0]]['functional_1']:
        print(hi)
    print()
    # print(loaded_weights[allKeys[0]]['conv1_1_W'])
    # print(loaded_weights[allKeys[0]]['conv1_1_W'][:])
    

def show_cropped_image(img_batch, mask_batch, img_path, img_format, strategy_type, save_path):
    print('show_cropped_image')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if 'normalize' in strategy_type:
        img_batch = np.moveaxis(img_batch, 1, -1) * 255  # axis move from first channel to last channel
    else:
        std, avg = get_std_mean_from_images(img_path, img_format)
        img_batch = np.moveaxis(img_batch, 1, -1) * std + avg
    mask_batch = np.moveaxis(mask_batch, 1, -1)*255
    print(img_batch.shape, mask_batch.shape)
    batch_size = img_batch.shape[0]

    color_index = 0
    # for color_index in [0,1,2]:
    for batch_index in range(100):
        cv2.imwrite(save_path + f'image_{batch_index}_{color_index}.png', img_batch[batch_index,:,:,color_index])
        cv2.imwrite(save_path + f'mask_{batch_index}_{color_index}.png', mask_batch[batch_index,:,:,color_index])
    # for batch_index in range(batch_size-100, batch_size):
    #     cv2.imwrite(save_path + f'image_{batch_index}_{color_index}.png', img_batch[batch_index,:,:,color_index])
    #     cv2.imwrite(save_path + f'mask_{batch_index}_{color_index}.png', mask_batch[batch_index,:,:,color_index])


def get_flops():
    # https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
    # https://github.com/tensorflow/tensorflow/issues/32809
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


# For training set, check if all the files have raw image and labeling
def check_namespace(img_path, msk_path, img_format):
    valid_list = []
    for file in self.namespace:
        if os.path.isfile(img_path + file) and os.path.isfile(msk_path + file) and file.endswith(img_format):
            valid_list.append(file)

    return valid_list


if __name__ == "__main__":
    save_path = 'results/debugger/'
    constants = UserParams('predict')
    frame = constants.frame_list[0]
    dataset_name = constants.dataset[0]

    temp_data = np.load(constants.get_crop_path() + dataset_name + '_' + str(frame) + '_split0_train_mask.npz')
    temp_img = temp_data['arr_0']
    temp_mask = temp_data['arr_1']
    
    print(temp_img.shape, temp_mask.shape)
    show_cropped_image(temp_img, temp_mask, dataset_name, save_path)