'''
Author Junbong Jang
Date 3/14/2021

fit_generator to lazy load images for training
referenced: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
'''

import sys

sys.path.append('..')
import numpy as np
import time
import os.path
import gc
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from deeplabv3 import Deeplabv3
from deep_neural_net import *
from deep_neural_net_3D import *
import loss
from debug_utils import *
from UserParams import UserParams
from custom_callback import TimeHistory
from train_data_generator import get_data_generators
from train_data_generator_3D import get_data_generators_3D


def build_model(constants, args, frame, model_name):
    pretrained_weights_path = constants.get_pretrained_weights_path(frame, model_name)
    if "Res50V2" in str(constants.strategy_type):
        model = ResNet50V2Keras(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "InceptionResV2" in str(constants.strategy_type):
        model = InceptionResV2(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                               weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "Dense201" in str(constants.strategy_type):
        model = DenseNet201Keras(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                 weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "deeplabv3" == str(constants.strategy_type):
        model = Deeplabv3(input_shape=(args.input_size, args.input_size, 3), output_shape=(68, 68), right_crop=0,
                          bottom_crop=0)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_dropout" in str(constants.strategy_type):
        model = VGG16_dropout(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_batchnorm" == str(constants.strategy_type):
        model = VGG16_batchnorm(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_instancenorm" == str(constants.strategy_type):
        model = VGG16_instancenorm(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                   weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_movie3" == str(constants.strategy_type):
        model = VGG16_movie(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                            weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=loss.temporal_cross_entropy, metrics=[loss.dice_coef])

    elif "VGG16_dice" == str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=[loss.dice_coef], metrics=['binary_crossentropy'])

    elif "VGG16_l2" == str(constants.strategy_type):
        model = VGG16_l2(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                         weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_dac_input256" == constants.strategy_type:
        model = VGG16_dac(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                          weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_spp_input256" == constants.strategy_type:
        model = VGG16_spp(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                          weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_no_pretrain" == str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path, encoder_weights=None)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16" in str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_attn_temporal" in str(constants.strategy_type):
        model = VGG19D_attn_temporal(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_dac_input256" in str(constants.strategy_type):
        model = VGG19_dropout_dac(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                  weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_feature_extractor" in str(constants.strategy_type):
        model = VGG19_dropout_feature_extractor(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                                weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy', loss.zero_loss],
                      metrics=[loss.dice_coef, loss.zero_loss])

    elif "VGG19_batchnorm_dropout" in str(constants.strategy_type):
        model = VGG19_batchnorm_dropout(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                        weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_gelu" in str(constants.strategy_type):
        model = VGG19_dropout_gelu(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                   weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_swish" in str(constants.strategy_type):
        model = VGG19_dropout_swish(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                    weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout" in str(constants.strategy_type):
        model = VGG19_dropout(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_batchnorm" == str(constants.strategy_type):
        model = VGG19_batchnorm(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_no_pretrain" == str(constants.strategy_type):
        model = VGG19(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path, encoder_weights=None)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19" in str(constants.strategy_type):
        model = VGG19(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "EFF_B7" == str(constants.strategy_type) or "EFF_B7_no_preprocessing" == str(constants.strategy_type):
        model = EFF_B7(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                       weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "unet_3D" in str(constants.strategy_type):
        model = UNet_3D(args.input_depth, args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                     weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "unet_feature_extractor" in str(constants.strategy_type):
        model = UNet_feature_extractor(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                       weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy', loss.zero_loss],
                      metrics=[loss.dice_coef, loss.zero_loss])

    elif "unet" in str(constants.strategy_type):
        model = UNet(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                     weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    return model


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

    if '_3D' in constants.strategy_type:
        train_x, train_y, valid_x, valid_y = get_data_generators_3D(train_val_dataset_names,
                        repeat_index, args.crop_mode, constants.img_format, process_type, args.input_depth)
    elif 'attn_temporal' in constants.strategy_type:
        aug_batch_size = 64
        train_x, train_y, valid_x, valid_y = get_data_generators(constants.round_num, train_val_dataset_names,
                    model_name, frame, repeat_index, args.crop_mode, constants.img_format, aug_batch_size, process_type, history_path)
        train_x = [train_x, train_x, train_x, train_x, train_x]
        valid_x = [valid_x, valid_x, valid_x, valid_x, valid_x]
    else:
        aug_batch_size = 64
        train_x, train_y, valid_x, valid_y = get_data_generators(constants.round_num, train_val_dataset_names,
                    model_name, frame, repeat_index, args.crop_mode, constants.img_format, aug_batch_size, process_type, history_path)

    if "deeplabv3" == str(constants.strategy_type) or "EFF_B7" == str(constants.strategy_type) or \
            "EFF_B7_no_preprocessing" == str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        # first channel to last channel
        train_x = np.moveaxis(train_x, 1, -1)
        train_y = np.moveaxis(train_y, 1, -1)
        valid_x = np.moveaxis(valid_x, 1, -1)
        valid_y = np.moveaxis(valid_y, 1, -1)

    # ------------ Build the model ------------
    model = build_model(constants, args, frame, model_name)

    # ------------ Sanity check the model ------------
    # print(model.summary())
    print('Num of layers: ', len(model.layers))
    # print('FLOPS: ', get_flops())  # run this after model compilation
    # check_loaded_weights(constants)
    if repeat_index == 0:
        plot_model(model, to_file='model_plots/model_round{}_{}_train.png'.format(constants.round_num,
                                                                                  constants.strategy_type),
                   show_shapes=True, show_layer_names=True, dpi=144)

    # ------------ Fit the Model ------------
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

    # reference https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/training.py#L1823-L1861
    hist = model.fit(train_x, train_y,
                     epochs=args.epochs,
                     verbose=1,
                     workers=1,
                     batch_size = args.train_batch_size,
                     validation_data=(valid_x, valid_y),
                     callbacks=[model_checkpoint, earlyStopping, time_callback, TensorBoard(log_dir=logdir)])

    # ------------ Save the History ------------
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
            for model_index in range(len(constants.model_names)):
                frame = constants.frame_list[frame_index]
                start_time = time.time()
                train_model(constants, model_index, frame, repeat_index, history_path)
                elapsed_time = time.time() - start_time
                print('Elapsed Time:', elapsed_time / 3600, 'hr')
                gc.collect()
