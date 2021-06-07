import sys
sys.path.append('..')

import gc
import numpy as np
import cv2
import os.path
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

import debug_utils
from predict_data_generator import PredictDataGenerator
from data_generator_classifier import get_data_generator_classifier
from UserParams import UserParams
from model_builder import build_model_predict


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

    if "classifier" in str(constants.strategy_type):
        input_images, mask_class_list = get_data_generator_classifier([dataset_name], repeat_index, 'even', constants.img_format, 'predict')
        image_rows, image_cols = input_images.shape[2:]
        orig_rows, orig_cols = 0, 0
    else:
        prediction_data_generator = PredictDataGenerator(img_path, mask_path, a_strategy, img_format=constants.img_format)
        input_images, image_filenames, image_cols, image_rows, orig_cols, orig_rows = prediction_data_generator.get_expanded_whole_frames()

        print('img size:', image_rows, image_cols)
        print('orig img size:', orig_rows, orig_cols)

    # ------------------- Load the trained Model -------------------
    model = build_model_predict(constants, frame, repeat_index, model_name, image_rows, image_cols, orig_rows, orig_cols)
    print('model layers: ', len(model.layers))
    plot_model(model, to_file='model_plots/model_round{}_{}_predict.png'.format(constants.round_num, constants.strategy_type), show_shapes=True, show_layer_names=True, dpi=144)

    # ------------------ Prediction and Save ------------------------------
    if "classifier" in str(constants.strategy_type):
        class_list_output = model.predict(input_images, batch_size=1, verbose=1)
        # model.evaluate(input_images, mask_class_list, batch_size=1, verbose=1)

        # thresholding prediction to calculate evaluation statistics
        class_list_output[class_list_output < 0.5] = 0
        class_list_output[class_list_output > 0] = 1

        y_pred = class_list_output[:, 0].tolist()
        y_true = mask_class_list

        from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, accuracy_score
        accuracy = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        mcc = matthews_corrcoef(y_true, y_pred)
        print(tn, fp, fn, tp, mcc, f1_score(y_true, y_pred), f1_score(y_true, y_pred, average='micro'), f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='weighted'), accuracy)

    else:
        if "feature_extractor" in str(constants.strategy_type):
            segmented_output, style_output = model.predict(input_images, batch_size = 1, verbose = 1)
            np.save(save_path + 'style_feature_vector.npy', style_output)

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

        for f in range(segmented_output.shape[0]):
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
