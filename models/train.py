'''
Author Junbong Jang
Date 3/14/2021

fit_generator to lazy load images for training
referenced: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
'''

import sys
sys.path.append('..')
sys.path.append('../data_handle')

import numpy as np
import time
import os.path
import gc
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

from debug_utils import *
from UserParams import UserParams
from custom_callback import TimeHistory
from model_builder import build_model_train
from data_generator_MTL import get_data_generator_MTL
from data_generator_classifier import get_data_generator_classifier
from train_data_generator import get_data_generator
from train_data_generator_3D import get_data_generator_3D, get_data_generator_3D_all


def train_model(constants, model_index, frame, repeat_index, history_path):
    model_name = constants.model_names[model_index]
    print(' round_num:', constants.round_num, ' model name:', model_name, ' frame:', frame, ' repeat_index:', repeat_index)
    args = constants.get_args()  # get hyper parameters

    # leave-one-movie-out cross validation so don't use the test movie
    train_val_dataset_names = [x for i, x in enumerate(constants.dataset_names) if i != model_index]
    print('train_val_dataset_names:', train_val_dataset_names)

    if 'paxillin_TIRF' in train_val_dataset_names[0] and \
        ('specialist' in constants.strategy_type or 'single_micro' in constants.strategy_type):
        process_type = 'normalize'
    else:
        process_type = 'standardize'

    # ---------------------- Load Data Generator --------------------------
    if '_classifier' in constants.strategy_type or '_MTL' in constants.strategy_type:
        train_x, train_y, valid_x, valid_y = get_data_generator_MTL(train_val_dataset_names, repeat_index, args.crop_mode, constants.img_format, 'train')
        if '_classifier' in constants.strategy_type and 'regressor' not in constants.strategy_type:
            # get mask class list only
            train_y = train_y[2]
            valid_y = valid_y[2]

    elif '_3D' in constants.strategy_type:
        train_x, train_y, valid_x, valid_y = get_data_generator_3D_all(train_val_dataset_names,
                        repeat_index, args.crop_mode, constants.img_format, process_type, args.input_depth)
    elif 'temporal' in constants.strategy_type:
        aug_batch_size = 64
        train_x, train_y, valid_x, valid_y = get_data_generator_3D(train_val_dataset_names, frame,
                        repeat_index, args.crop_mode, constants.img_format, aug_batch_size, process_type)
    else:
        aug_batch_size = 64
        train_x, train_y, valid_x, valid_y = get_data_generator(constants.round_num, train_val_dataset_names,
                    model_name, frame, repeat_index, args.crop_mode, constants.img_format, aug_batch_size, process_type, history_path)

    if "deeplabv3" == str(constants.strategy_type) or "EFF_B" in str(constants.strategy_type) \
            or 'imagenet_pretrained' in str(constants.strategy_type)\
            or 'vit_classifier' in str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        # first channel to last channel
        train_x = np.moveaxis(train_x, 1, -1)
        valid_x = np.moveaxis(valid_x, 1, -1)
        if 'classifier' not in str(constants.strategy_type):
            train_y = np.moveaxis(train_y, 1, -1)
            valid_y = np.moveaxis(valid_y, 1, -1)
    print('train_x', train_x.shape, 'valid_x', valid_x.shape)
    
    # ---------------------- Build the model ----------------------
    # multiple gpu training
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = build_model_train(constants, args, frame, model_name)

    # ---------------------- Sanity Check the model ----------------------
    print(model.summary())
    print('Num of layers: ', len(model.layers))
    # print('FLOPS: ', get_flops())  # run this after model compilation
    # check_loaded_weights(constants)
    if repeat_index == 0:
        plot_model(model, to_file='model_plots/model_round{}_{}_train.png'.format(constants.round_num,
                                                                                  constants.strategy_type),
                   show_shapes=True, show_layer_names=True, dpi=144)

    # ---------------------- Fit the Model ----------------------
    print('Fit Model...', args.epochs, args.patience)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=0, mode='auto')  # args.patience
    model_checkpoint = ModelCheckpoint(
        'results/model_round{}_{}/model_frame{}_{}_repeat{}.hdf5'.format(constants.round_num, constants.strategy_type,
                                                                         str(frame), model_name,
                                                                         str(repeat_index)),
        monitor='val_loss', save_best_only=True)

    time_callback = TimeHistory()
    logdir = 'results/history_round{}_{}/tensorboard_frame{}_{}_repeat{}_{}'.format(constants.round_num,
                                                                                constants.strategy_type, str(frame),
                                                                                model_name,
                                                                                str(repeat_index),
                                                        datetime.now().strftime("%Y%m%d-%H%M%S"))

    if 'classifier_regressor' in constants.strategy_type:
        print(train_y[0].shape, train_y[1].shape, train_y[2].shape)
        hist = model.fit(train_x, [train_y[1], train_y[2]],
                         epochs=args.epochs,
                         verbose=1,
                         workers=1,
                         batch_size = args.train_batch_size,
                         validation_data=(valid_x, [valid_y[1], valid_y[2]]),
                         callbacks=[model_checkpoint, earlyStopping, time_callback, TensorBoard(log_dir=logdir)])
    else:
        # reference https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/training.py#L1823-L1861
        hist = model.fit(train_x, train_y,
                         epochs=args.epochs,
                         verbose=1,
                         workers=1,
                         batch_size = args.train_batch_size,
                         validation_data=(valid_x, valid_y),
                         callbacks=[model_checkpoint, earlyStopping, time_callback, TensorBoard(log_dir=logdir)])

    # ----------------------  Save the Training History ----------------------
    hist.history['times'] = time_callback.times
    print('Save History...')
    np.save('results/history_round{}_{}/history_frame{}_{}_repeat{}.npy'.format(constants.round_num,
                                                                                constants.strategy_type, str(frame),
                                                                                model_name,
                                                                                str(repeat_index)), hist.history)
    K.clear_session()

    return


if __name__ == "__main__":

    K.set_image_data_format('channels_first')
    print(K.image_data_format())
    constants = UserParams('train')

    history_path = 'results/history_round{}_{}/'.format(constants.round_num, constants.strategy_type)
    print(history_path)
    if not os.path.exists(history_path):
        os.makedirs(history_path)
    if not os.path.exists('results/model_round{}_{}/'.format(constants.round_num, constants.strategy_type)):
        os.makedirs('results/model_round{}_{}/'.format(constants.round_num, constants.strategy_type))
    for repeat_index in range(constants.REPEAT_MAX):
        for frame_index in range(len(constants.frame_list)):
            for model_index in range(0,len(constants.model_names)):
                frame = constants.frame_list[frame_index]
                start_time = time.time()
                train_model(constants, model_index, frame, repeat_index, history_path)
                elapsed_time = time.time() - start_time
                print('Elapsed Time:', elapsed_time / 3600, 'hr')
                gc.collect()
