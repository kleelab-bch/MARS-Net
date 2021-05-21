import sys
sys.path.append('..')

import gc
import numpy as np
import cv2
import os.path
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

import debug_utils
from deeplabv3 import Deeplabv3
from deep_neural_net import *
from deep_neural_net_3D import *
from predict_data_generator import PredictDataGenerator
from UserParams import UserParams


def build_model(constants, frame, model_name, image_rows, image_cols, orig_rows, orig_cols):
    weights_path = constants.get_trained_weights_path(str(frame), model_name, str(repeat_index))

    if "Res50V2" in str(constants.strategy_type):
        model = ResNet50V2Keras(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "Dense201" == str(constants.strategy_type):
        model = DenseNet201Keras(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "InceptionResV2" == str(constants.strategy_type):
        model = InceptionResV2(image_rows, image_cols, 0, image_cols - orig_cols, image_rows - orig_rows, weights_path=weights_path)
    elif "deeplabv3" == str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        input_images = np.moveaxis(input_images, 1, -1)  # first channel to last channel
        print(input_images.dtype, input_images.shape)
        model = Deeplabv3(input_shape=(image_rows, image_cols, 3), output_shape=(orig_rows, orig_cols))
        model.load_weights(weights_path, by_name=True)

    elif "VGG16_dropout" == str(constants.strategy_type):
        model = VGG16_dropout(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16_batchnorm" == str(constants.strategy_type):
        model = VGG16_batchnorm(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16_instancenorm" == str(constants.strategy_type):
        model = VGG16_instancenorm(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "movie3" in str(constants.strategy_type):
        model = VGG16_movie(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16_dac_input256" == constants.strategy_type:
        model = VGG16_dac(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16_spp_input256" == constants.strategy_type:
        model = VGG16_spp(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16" in str(constants.strategy_type):
        model = VGG16(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)

    elif "VGG19D_attn_temporal" in str(constants.strategy_type):
        model = VGG19D_attn_temporal(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout_gelu" in str(constants.strategy_type):
        model = VGG19_dropout_gelu(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout_swish" in str(constants.strategy_type):
        model = VGG19_dropout_swish(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout_dac" in str(constants.strategy_type):
        model = VGG19_dropout_dac(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout_feature_extractor" in str(constants.strategy_type):
        model = VGG19_dropout_feature_extractor(image_rows, image_cols, 0, image_cols - orig_cols, image_rows - orig_rows, weights_path=weights_path)
    elif "VGG19_batchnorm_dropout" in str(constants.strategy_type):
        model = VGG19_batchnorm_dropout(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_batchnorm" in str(constants.strategy_type):
        model = VGG19_batchnorm(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout" in str(constants.strategy_type):
        model = VGG19_dropout(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19" in str(constants.strategy_type):
        model = VGG19(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path, encoder_weights=None)

    elif "EFF_B7" in str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        input_images = np.moveaxis(input_images, 1, -1)  # first channel to last channel
        print(input_images.dtype, input_images.shape)
        model = EFF_B7(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)

    elif "unet_3D" in str(constants.strategy_type):
        model = UNet_3D(32, image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "unet_feature_extractor" in str(constants.strategy_type):
        model = UNet_feature_extractor(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "unet" in str(constants.strategy_type):
        model = UNet(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)

    return model


def prediction(constants, frame, model_index, repeat_index, save_path):
    model_name = constants.model_names[model_index]
    dataset_folder = constants.dataset_folders[model_index]
    dataset_name = constants.dataset_names[model_index]
    img_folder = constants.img_folders[model_index]
    mask_folder = constants.mask_folders[model_index]

    img_path = dataset_folder + dataset_name + img_folder
    mask_path = dataset_folder + dataset_name + mask_folder
    
    if constants.self_training_type is None:
        save_path = save_path + '{}/frame{}_{}_repeat{}/'.format(dataset_name, str(frame), model_name, str(repeat_index))
    else:
        save_path = save_path + '{}_{}/frame{}_repeat{}/'.format(model_name, dataset_name, str(frame), str(repeat_index))
    print('save_path:', save_path)
    if os.path.isdir(save_path) == 0:
        os.makedirs(save_path)

    # ------------------- Data loading -------------------
    a_strategy = constants.strategy_type
    if 'TIRF' in dataset_name and 'specialist' in constants.strategy_type:
        a_strategy = constants.strategy_type + '_normalize'

    prediction_data_generator = PredictDataGenerator(img_path, mask_path, a_strategy, img_format=constants.img_format)
    input_images, image_filenames, image_cols, image_rows, orig_cols, orig_rows = prediction_data_generator.get_expanded_whole_frames()

    print('img size:', image_rows, image_cols)
    print('orig img size:', orig_rows, orig_cols)

    # ------------------- Load the trained Model -------------------
    model = build_model(constants, frame, model_name, image_rows, image_cols, orig_rows, orig_cols)
    print('model layers: ', len(model.layers))
    plot_model(model, to_file='model_plots/model_round{}_{}_predict.png'.format(constants.round_num, constants.strategy_type), show_shapes=True, show_layer_names=True, dpi=144)

    # ------------------- predict segmented images and save them -------------------

    if "feature_extractor" in str(constants.strategy_type):
        segmented_output, style_output = model.predict(input_images, batch_size = 1, verbose = 1)
        np.save(save_path + 'style_feature_vector.npy', style_output)

    elif "attn_temporal" in str(constants.strategy_type):
        input_images = [input_images, input_images, input_images, input_images, input_images]
        segmented_output = model.predict(input_images, batch_size = 1, verbose = 1)

    elif "_3D" in str(constants.strategy_type):
        input_images = input_images[np.newaxis,:32,:,:,:]  # image shape: depth, channel, width, height
        input_images = np.moveaxis(input_images, 1, 2)  # new image shape: 1, channel, depth, width, height
        segmented_output = model.predict(input_images, batch_size = 1, verbose = 1) # output shape (1, 1, 16, 474, 392)
        segmented_output = np.moveaxis(segmented_output[0], 0, 1)  # new output shape: (16, 1, 474, 392)
    else:
        segmented_output = model.predict(input_images, batch_size = 1, verbose = 1)

    segmented_output = 255 * segmented_output  # 0=black color and 255=white color

    if "deeplabv3" == str(constants.strategy_type) or "EFF_B7" == str(constants.strategy_type) or "EFF_B7_no_preprocessing" == str(constants.strategy_type):
        # move last channel to first channel
        segmented_output = np.moveaxis(segmented_output, -1, 1)
        print(segmented_output.shape)

    for f in range(len(image_filenames)):
        if constants.strategy_type == 'movie3' or constants.strategy_type == 'movie3_loss':
            out = segmented_output[f, 1, :, :]
        else:
            out = segmented_output[f, 0, :, :]
        cv2.imwrite(save_path + image_filenames[f], out)
    K.clear_session()


def get_dataset_name(constants):
    if 'one_generalist' in str(constants.strategy_type):
        dataset_names = constants.dataset_names  # one model predicts all datasets
    elif constants.self_training_type is None or constants.round_num == constants.final_round_num:  # assume round 2 is the last
        # In Cross Validation, one model predicts one test dataset
        dataset_names = [constants.dataset_names[model_index]]
    else:  # In self training, one model trained on 4 datasets predicts those same 4 datasets
        dataset_names = constants.get_datasets_for_model(model_name)


if __name__ == "__main__":
    K.set_image_data_format('channels_first')
    constants = UserParams('predict')

    root_prediciton_path = "results/predict_wholeframe_round{}_{}/".format(constants.round_num, constants.strategy_type)

    if len(constants.model_names) != 1 and len(constants.dataset_names) != len(constants.model_names):
        raise Exception('Length of Dataset names and Model names are not the same')

    # for self training, ABCD model predicts for dataset A,B,C,D
    # for test set prediction, ABCD model predicts the dataset E
    for repeat_index in range(constants.REPEAT_MAX):
        for frame in constants.frame_list:
            for model_index in range(len(constants.model_names)):
                prediction(constants, frame, model_index, repeat_index, root_prediciton_path)
            gc.collect()
