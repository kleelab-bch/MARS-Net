'''
Author Junbong Jang
Date 7/15/2021

To evaluate classification results with respect to the ground truth label
'''

import sys
sys.path.append('..')
import numpy as np
from UserParams import UserParams
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score


def eval_summary(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    mcc = matthews_corrcoef(y_true, y_pred)
    print('classification tn,fp,fn,tp', round(tn, 4), round(fp, 4), round(fn, 4), round(tp, 4),
          '\nmcc', round(mcc, 4),
          '\nprecision', round(precision_score(y_true, y_pred), 4),
          '\nrecall', round(recall_score(y_true, y_pred), 4),
          '\nf1', round(f1_score(y_true, y_pred), 4),
          '\naccuracy', round(accuracy, 4),
          '\n-----------------------------')


def get_pred_results(save_path):
    y_pred = np.load(save_path + 'class_list_pred.npy', allow_pickle=True)
    y_true = np.load(save_path + 'class_list_true.npy', allow_pickle=True)

    prediction_threshold = 0.5
    y_pred[y_pred < prediction_threshold] = 0
    y_pred[y_pred > 0] = 1
    y_pred = y_pred[:,0]

    return y_true, y_pred


def convert_classified_grids_to_boxes(classified_list, save_path, box_save_name):
    # Convert classified grids of the image into bounding box coordinates
    print('convert_classified_grids_to_boxes', box_save_name, len(classified_list))

    # assume images are cropped into 256x256 patches with 50% overlap and there are 432 patches in total (24x18)
    # remember that images are first cropped from top to bottom, and then left to right
    num_patches_per_img = 432
    width_patches = 24
    height_pathces = 18
    assert num_patches_per_img == width_patches*height_pathces

    img_height = 1944
    img_width = 2592
    patch_size = 256
    crop_patch_offset = (patch_size-60) * 0.5
    assert img_width%width_patches == 0
    assert img_height%height_pathces == 0

    boxes_per_image = {}

    for img_index in range(41):  # test set has 41 images
        xmins = np.array([])
        ymins = np.array([])
        xmaxs = np.array([])
        ymaxs = np.array([])

        classified_count = 0
        classified_patches = classified_list[num_patches_per_img*img_index:num_patches_per_img*(img_index+1)]
        for patch_index, classified_patch in enumerate(classified_patches):
            height_coord = patch_index % height_pathces
            width_coord = patch_index // height_pathces
            # print(height_coord, width_coord)
            xmin = crop_patch_offset * width_coord
            ymin = crop_patch_offset * height_coord
            xmax = xmin + patch_size
            ymax = ymin + patch_size

            if classified_patch == 1:
                classified_count = classified_count + 1
                xmins = np.append(xmins, xmin)
                ymins = np.append(ymins, ymin)
                xmaxs = np.append(xmaxs, xmax)
                ymaxs = np.append(ymaxs, ymax)
            # print(xmin, ymin, xmax, ymax)
        print('classified_count', classified_count)
        # reshape (n,) vector into (nx1) column vector
        xmins = np.expand_dims(xmins, 1)
        ymins = np.expand_dims(ymins, 1)
        xmaxs = np.expand_dims(xmaxs, 1)
        ymaxs = np.expand_dims(ymaxs, 1)

        # bounding boxes in nx4 matrix
        bounding_boxes = np.concatenate((ymins, xmins, ymaxs, xmaxs), axis=1)

        boxes_per_image[img_index] = bounding_boxes

    np.save(save_path + f'{box_save_name}_boxes.npy', boxes_per_image)


if __name__ == "__main__":
    constants = UserParams('predict')

    if len(constants.model_names) != 1 and len(constants.dataset_names) != len(constants.model_names):
        raise Exception('Length of Dataset names and Model names are not the same')

    repeat_index = 0
    frame = 2
    for model_index in range(len(constants.model_names)):
        model_name = constants.model_names[model_index]
        dataset_name = constants.dataset_names[model_index]
        save_path = constants.get_save_prediction_path(dataset_name, model_name, frame, repeat_index)

        y_true, y_pred = get_pred_results(save_path)

        convert_classified_grids_to_boxes(y_pred, save_path, 'pred')
        convert_classified_grids_to_boxes(y_true, save_path, 'ground_truth')

        if model_index == 0:
            y_true_all = y_true
            y_pred_all = y_pred
        else:
            y_true_all = np.concatenate((y_true_all, y_true))
            y_pred_all = np.concatenate((y_pred_all, y_pred))

    print(y_true_all.shape, y_pred_all.shape)
    eval_summary(y_true_all, y_pred_all)

