import sys
sys.path.append('..')
sys.path.append('../models/')

import gc
import numpy as np
import cv2
import os
from tensorflow.keras import backend as K

from deep_neural_net import *
from data_processor import aggregate_std_mean, get_std_mean_from_images
from predict_data_generator import DataGenerator
from UserParams import UserParams

from seggradcam import SegGradCAM, BiasRoI, SuperRoI, ClassRoI, PixelRoI
from visualize_sgc import SegGradCAMplot


def run_seggradcam(constants, dataset_folder, mask_folder, img_folder, model_name, dataset_name, frame, repeat_index, root_path):

    img_path = dataset_folder + dataset_name + constants.img_folder
    mask_path = dataset_folder + dataset_name + constants.get_mask_folder(model_name, dataset_name, frame, 0)
    print(img_path)
    print(mask_path)

    if constants.self_training_type is None:
        save_path = root_path + '{}/frame{}_{}_repeat{}/'.format(dataset_name, str(frame), model_name, str(repeat_index))
    else:
        save_path = root_path + '{}_{}/frame{}_repeat{}/'.format(model_name, dataset_name, str(frame), str(repeat_index))
    print('save_path:', save_path)
    if os.path.isdir(save_path) == 0:
        os.makedirs(save_path)

    # ------------------- Data loading -------------------
    temp_strategy_type = constants.strategy_type
    if 'TIRF' in dataset_name and 'specialist' in constants.strategy_type:
        temp_strategy_type = constants.strategy_type + '_normalize'

    prediction_data_generator = PredictDataGenerator(img_path, mask_path, temp_strategy_type, img_format=constants.img_format)
    expanded_images, namelist, image_cols, image_rows, orig_cols, orig_rows = prediction_data_generator.get_expanded_whole_frames()
    orig_images, _, _, _ = prediction_data_generator.get_orig_whole_frames()
    masks = prediction_data_generator.get_mask_frames()

    print('img size:', image_rows, image_cols)
    print('orig img size:', orig_rows, orig_cols)
    print('expanded_images: ', expanded_images.dtype, expanded_images.shape)
    print('masks: ', masks.shape)

    # ------------------- Model Loading -------------------
    weights_path = constants.get_trained_weights_path(str(frame), model_name, str(repeat_index), '../models/')

    if "VGG19_dropout" in str(constants.strategy_type):
        model = VGG19_dropout(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
        prop_to_layer_list = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
        # ['conv2d_1', 'conv2d_3', 'conv2d_5', 'conv2d_7']

    elif "unet" in str(constants.strategy_type):
        model = UNet(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
        prop_to_layer_list = ['conv2d_1','conv2d_3','conv2d_5','conv2d_7','conv2d_9']

    pred_images = model.predict(expanded_images, batch_size = 1, verbose = 1)
    print('pred_images: ', pred_images.shape)

    # ------------------- SegGradCAM -------------------
    prop_from_layer = model.layers[-1].name  # crop layer
    cls = 0  # class

    # per layer
    for prop_to_layer in prop_to_layer_list:
        print('prop_to_layer: ', prop_to_layer)
        for img_index in range(0,3): # expanded_images.shape[0]
            orig_image = orig_images[img_index,:,:,:]
            expanded_image = expanded_images[img_index,:,:,:]

            pred_image = pred_images[img_index,0,:,:]
            edge_pred = extract_edge(pred_image)

            ground_truth_mask = masks[img_index,:,:]
            edge_gt = extract_edge(ground_truth_mask)

            # ---------- For pixel
            # pixsgc = SegGradCAM(model, expanded_image, pred_image, cls, prop_to_layer, prop_from_layer, roi=PixelRoI(33, 29, pred_image),
            #                     normalize=True, abs_w=False, posit_w=False)
            # pixsgc.SGC()
            # plotter = SegGradCAMplot(pixsgc, next_dict=None, image_id=img_index, gt=None, n_classes=1, outfolder=save_path)
            # plotter.explainPixel()

            # ---------- For class
            classgc = SegGradCAM(model, expanded_image, edge_gt, cls, prop_to_layer, prop_from_layer, roi=ClassRoI(edge_gt, orig_image, cls), abs_w=False, posit_w=False)
            classgc.SGC()

            print('cam mean: ', np.mean(classgc.cam), ' max: ', classgc.cam.max())
            classgc.cam = classgc.cam / classgc.cam.max()  # normalize non-negative weighted sum

            clsplotter = SegGradCAMplot(classgc, next_dict=None, image_id=img_index, image_name=namelist[img_index].replace('.png',''),
                                        ground_truth_mask=edge_gt, n_classes=1, outfolder=save_path) # , pred_image=edge_pred[0]
            clsplotter.explainClass()


    # combine cam from all feature layers
    # for img_index in range(0,3):
    #     total_cam = 0
    #     orig_image = orig_images[img_index,:,:,:]
    #     expanded_image = expanded_images[img_index,:,:,:]
    #
    #     pred_image = pred_images[img_index,0,:,:]
    #     edge_pred = extract_edge(pred_image)
    #
    #     ground_truth_mask = masks[img_index,:,:]
    #     edge_gt = extract_edge(ground_truth_mask)
    #
    #     for prop_to_layer in prop_to_layer_list:
    #         print(prop_to_layer)
    #         classgc = SegGradCAM(model, expanded_image, edge_gt, cls, prop_to_layer, prop_from_layer, roi=ClassRoI(edge_gt, orig_image, cls),
    #                             abs_w=False, posit_w=False)
    #         classgc.SGC()
    #         total_cam = total_cam + classgc.cam
    #         print('classgc.cam mean: ', np.mean(classgc.cam), ' max: ', classgc.cam.max())
    #
    #     print('total_cam mean: ', np.mean(total_cam), ' max: ', total_cam.max())
    #     # normalize non-negative weighted sum
    #     total_cam = total_cam / total_cam.max()
    #     print('total_cam mean: ', np.mean(total_cam), ' max: ', total_cam.max())
    #
    #     # combining cam
    #     classgc.cam = total_cam
    #     classgc.prop_to_layer = 'combined'
    #     clsplotter = SegGradCAMplot(classgc, next_dict=None, image_id=img_index, image_name=namelist[img_index].replace('.png',''),
    #                                 ground_truth_mask=edge_gt, pred_image=edge_pred[0], n_classes=1, outfolder=save_path)
    #     clsplotter.explainClass()

    K.clear_session()


def extract_edge(an_image):
    an_image = an_image * 255
    an_image = an_image.astype(np.uint8)
    edge_image = cv2.Canny(an_image, 100, 200) / 255
    edge_image = np.expand_dims(edge_image, axis=0)

    return edge_image


if __name__ == "__main__":
    K.set_image_data_format('channels_first')
    constants = UserParams('predict')
    root_path = "results/round{}_{}/".format(constants.round_num, constants.strategy_type)

    if len(constants.model_names) != 1 and len(constants.dataset_names) != len(constants.model_names):
        raise Exception('Length of Dataset names and Model names are not the same')

    for repeat_index in range(0,1):# constants.REPEAT_MAX
        for model_name, dataset_folder, dataset_name, mask_folder, img_folder in zip(constants.model_names, constants.dataset_folders, constants.dataset_names, constants.mask_folders, constants.img_folders):
            for frame in constants.frame_list:
                run_seggradcam(constants, dataset_folder, mask_folder, img_folder, model_name, dataset_name, frame, repeat_index, root_path)

            gc.collect()
