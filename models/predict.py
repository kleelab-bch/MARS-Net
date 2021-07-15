import sys
sys.path.append('..')
sys.path.append('../data_handle')

import gc
import numpy as np
import cv2
import os.path
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model

import debug_utils
from predict_data_generator import PredictDataGenerator
from data_generator_classifier import get_data_generator_classifier, threshold_mask_area_list
from data_generator_MTL import get_data_generator_MTL
from UserParams import UserParams
from model_builder import build_model_predict
from visualization.CAM import visualize_feature_activation_map
from tqdm import tqdm


def prediction(constants, frame, model_index, repeat_index):
    model_name = constants.model_names[model_index]
    dataset_folder = constants.dataset_folders[model_index]
    dataset_name = constants.dataset_names[model_index]
    img_folder = constants.img_folders[model_index]
    mask_folder = constants.mask_folders[model_index]

    img_path = dataset_folder + dataset_name + img_folder
    mask_path = dataset_folder + dataset_name + mask_folder
    args = constants.get_args()  # get hyper parameters

    save_path = constants.get_save_prediction_path(dataset_name, model_name, frame, repeat_index)

    # ------------------- Data loading -------------------
    a_strategy = constants.strategy_type
    if 'TIRF' in dataset_name and 'specialist' in constants.strategy_type:
        a_strategy = constants.strategy_type + '_normalize'

    if "classifier" in str(constants.strategy_type) or 'MTL' in str(constants.strategy_type):
        orig_input_images, input_images, masks, image_filenames = get_data_generator_MTL([dataset_name], repeat_index, args.crop_mode, constants.img_format, 'predict')
        masks, mask_area_list, mask_classes = masks[0], masks[1], masks[2]
        image_rows, image_cols = input_images.shape[2:]
        orig_rows, orig_cols = 0, 0
    
    else:
        prediction_data_generator = PredictDataGenerator(img_path, mask_path, a_strategy, img_format=constants.img_format)
        input_images, image_filenames, image_cols, image_rows, orig_cols, orig_rows = prediction_data_generator.get_expanded_whole_frames()

        print('img size:', image_rows, image_cols)
        print('orig img size:', orig_rows, orig_cols)

    if "deeplabv3" == str(constants.strategy_type) or "EFF_B" in str(constants.strategy_type) \
        or "imagenet_pretrained" in str(constants.strategy_type) \
        or "vit_classifier" in str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        input_images = np.moveaxis(input_images, 1, -1)  # first channel to last channel
        print('input_images', input_images.dtype, input_images.shape)

    # ------------------- Load the trained Model -------------------
    model = build_model_predict(constants, frame, repeat_index, model_name, image_rows, image_cols, orig_rows, orig_cols)
    print('model layers: ', len(model.layers))
    plot_model(model, to_file='model_plots/model_round{}_{}_predict.png'.format(constants.round_num, constants.strategy_type), show_shapes=True, show_layer_names=True, dpi=144)

    # ------------------ Prediction and Save ------------------------------

    if "classifier" in str(constants.strategy_type) or "MTL" in str(constants.strategy_type):
        from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
        if "classifier" in str(constants.strategy_type) and 'regressor' not in str(constants.strategy_type)\
                and 'vit_classifier' not in str(constants.strategy_type):
            model = Model(inputs=model.input, outputs=[model.output, model.get_layer('global_average_pooling2d').output])
            pred_class_list, encoded_feature_vector = model.predict(input_images, batch_size=1, verbose=1)
            print('pred_class_list', pred_class_list.shape, encoded_feature_vector.shape, pred_class_list[:100, 0])
            np.save(save_path + 'class_list_pred.npy', pred_class_list)
            np.save(save_path + 'feature_vector.npy', encoded_feature_vector)
        elif 'vit_classifier' in str(constants.strategy_type):
            pred_class_list = model.predict(input_images, batch_size=1, verbose=1)
            np.save(save_path + 'class_list_pred.npy', pred_class_list)
        else:
            model = Model(inputs=model.input, outputs=model.output + [model.get_layer('global_average_pooling2d').output])
            if '_MTL' in str(constants.strategy_type):
                pred_mask, pred_mask_area_list, pred_class_list, encoded_feature_vector = model.predict(input_images, batch_size=1, verbose=1)
            else:
                pred_mask_area_list, pred_class_list, encoded_feature_vector = model.predict(input_images, batch_size=1, verbose=1)
            print(pred_mask_area_list.shape, pred_class_list.shape, encoded_feature_vector.shape)
            np.save(save_path + 'mask_area_list.npy', pred_mask_area_list)
            print('regression:', mean_squared_error(mask_area_list, pred_mask_area_list))
            print('pred_mask_area_list', pred_mask_area_list[pred_mask_area_list>1])

            np.save(save_path + 'class_list_pred.npy', pred_class_list)
            np.save(save_path + 'feature_vector.npy', encoded_feature_vector)

        # thresholding prediction to calculate evaluation statistics
        prediction_threshold = 0.5
        pred_class_list[pred_class_list < prediction_threshold] = 0
        pred_class_list[pred_class_list > 0] = 1

        y_pred = pred_class_list[:, 0].tolist()
        y_true = mask_classes
        np.save(save_path + 'class_list_true.npy', y_true)
        print(y_pred[:10])
        print(y_true[:10])

        accuracy = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        mcc = matthews_corrcoef(y_true, y_pred)
        print('classification tn,fp,fn,tp', round(tn, 4), round(fp, 4), round(fn, 4), round(tp, 4),
              '\nmcc',round(mcc, 4),
              '\nprecision',round(precision_score(y_true, y_pred), 4),
              '\nrecall',round(recall_score(y_true, y_pred), 4),
              '\nf1',round(f1_score(y_true, y_pred), 4),
              '\naccuracy', round(accuracy, 4),
              '\n-----------------------------')

        # save images if fp, fn or tp
        prediction_result_list = []
        for i in tqdm(range(len(y_true))):
            if y_true[i] == 1 and y_pred[i] == 0:
                prefix = 'FN'
            elif y_true[i] == 1 and y_pred[i] == 1:
                prefix = 'TP'
            elif y_true[i] == 0 and y_pred[i] == 1:
                prefix = 'FP'
            elif y_true[i] == 0 and y_pred[i] == 0:
                prefix = 'TN'
            if prefix != 'TN':
                image_filename = image_filenames[i].split('/')[-1]
                # cv2.imwrite(save_path + f'{prefix}_' + image_filename, np.moveaxis(orig_input_images[i], 0, -1))

            prediction_result_list.append(prefix)

        np.save(save_path + 'prediction_result_list.npy', prediction_result_list)

    # For segmentation outputs
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

        if "deeplabv3" == str(constants.strategy_type) or "EFF_B" in str(constants.strategy_type) or "imagenet_pretrained" in str(constants.strategy_type):
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

    if len(constants.model_names) != 1 and len(constants.dataset_names) != len(constants.model_names):
        raise Exception('Length of Dataset names and Model names are not the same')

    # for self training, ABCD model predicts for dataset A,B,C,D
    # for test set prediction, ABCD model predicts the dataset E
    for repeat_index in range(constants.REPEAT_MAX):
        for frame in constants.frame_list:
            for model_index in range(1): # len(constants.model_names)
                prediction(constants, frame, model_index, repeat_index)
            gc.collect()
