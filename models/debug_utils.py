'''
Author Junbong Jang
Date 9/2/2020

Contains debugging functions useful for deep learning research

'''
import sys
sys.path.append('..')
from UserParams import UserParams
from data_processor import get_std_mean_from_images

import math
import os
import cv2
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
from functools import wraps


def log_function_call(a_func):
    """decorator to log which function was called"""
    @wraps(a_func)
    def wrapTheFunction(*args, **kwargs):
        print(a_func.__name__, "is called @@@@@@@@@@@@@@@@@@")
        return a_func(*args, **kwargs)

    return wrapTheFunction


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


def calc_receptive_field(n_in, jump_in, r_in, center_in, orig_img_size, padding, kernel_size, stride):
    # refered to https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    n_out = math.floor((n_in + 2*padding - kernel_size)/stride) + 1  # number of features in one dimension
    jump_out = jump_in * stride  # distance between two consecutive feature
    r_out = r_in + (kernel_size - 1) * jump_in  # receptive field

    # center coordinate of the first or last feature
    if center_in > orig_img_size / 2:
        center_out = center_in - ((kernel_size-1)/2.0 - padding) * jump_in
    elif center_in < orig_img_size / 2:
        center_out = center_in + ((kernel_size-1)/2.0 - padding) * jump_in
    else:
        center_out = center_in
    return n_out, jump_out, r_out, center_out


if __name__ == "__main__":
    # save_path = 'results/debugger/'
    # constants = UserParams('predict')
    # frame = constants.frame_list[0]
    # dataset_name = constants.dataset[0]
    #
    # temp_data = np.load(constants.get_crop_path() + dataset_name + '_' + str(frame) + '_split0_train_mask.npz')
    # temp_img = temp_data['arr_0']
    # temp_mask = temp_data['arr_1']
    #
    # print(temp_img.shape, temp_mask.shape)
    # show_cropped_image(temp_img, temp_mask, dataset_name, save_path)

    # ---------------------------------------
    r_in = 1
    jump_in = 1
    n_in = 128
    orig_img_size = n_in
    center_in = 0.5
    for layer_type in ['conv','conv','maxpool',
                       'conv','conv','maxpool',
                       'conv','conv','conv','conv','maxpool',
                       'conv','conv','conv','conv','maxpool',
                       'conv','conv','conv','conv']:
        if layer_type == 'conv':
            kernel_size = 3
            stride = 1
            padding = 1
        elif layer_type == 'maxpool':
            kernel_size = 2
            stride = 2
            padding = 0

        n_in, jump_in, r_in, center_in = calc_receptive_field(n_in, jump_in, r_in, center_in, orig_img_size, padding, kernel_size, stride)
        print(layer_type, 'n:', n_in, '  jump:', jump_in, '  r:', r_in, '  center_in:', center_in)
