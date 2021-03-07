# Author Junbong Jang
# Date: 12/21/2020

# process images and masks by rotating and cropping them
# before they are used for training the neural network or windowing

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import os
from PIL import Image


def find_namespace(img_path, img_format):
    namespace = []

    img_file_name = os.listdir(img_path)
    for file in img_file_name:
        if os.path.isfile(img_path + file) and file.endswith(img_format):
            namespace.append(file)
    return namespace


def crop_images(img_path, img_format, img_type, save_path, expand_rotate, left, top, right, bottom, rotate_angle_clockwise):
    image_names = find_namespace(img_path, img_format)
    print(left, top, right, bottom)
    for image_name in image_names:
        a_image = Image.open(img_path + image_name)
        width, height = a_image.size

        if right == -1:
            right = width
        if bottom == -1:
            bottom = height
        if img_type == 'mask':
            a_image = a_image.convert('1')
        a_image = a_image.rotate(-1*rotate_angle_clockwise, expand=expand_rotate)
        a_image = a_image.crop((left, top, right, bottom))
        a_image.save(save_path + image_name)


def crop_for_windowing():
    # strategy_type = 'generalist_VGG19_dropout'
    strategy_type = 'generalist_unet'

    # dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
    #                  '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
    #                  '040119_PtK1_S01_01_phase']
    # model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']

    # dataset_names = ['Paxilin-HaloTMR-TIRF3','Paxilin-HaloTMR-TIRF4','Paxilin-HaloTMR-TIRF4','Paxilin-HaloTMR-TIRF7','Paxilin-HaloTMR-TIRF8']
    # model_names = ['K','L','N','O']
    dataset_names = ['Paxilin-HaloTMR-TIRF3']
    model_names = ['K']
    frame_num = 2

    # strategy_type = 'paxillin_TIRF_normalize_cropped_VGG19_dropout_patience_10'
    # dataset_names = ['Paxilin-HaloTMR-TIRF5']
    # model_names = ['ABCEF']
    # frame_num = 22

    img_format = '.png'
    img_folder = 'img'

    for dataset_index in range(len(dataset_names)):
        print('dataset:', dataset_names[dataset_index])
        model_predict_path = f"../models/results/predict_wholeframe_round1_{strategy_type}/{dataset_names[dataset_index]}/processed_frame{frame_num}_{model_names[dataset_index]}_repeat0/"

        model_save_path = f"generated/{dataset_names[dataset_index]}/predict_{strategy_type}/"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        expand_rotate = 0
        rotate_angle_clockwise = 0

        if dataset_names[dataset_index] == '040119_PtK1_S01_01_phase_3_DMSO_nd_03':
            left, top, right, bottom = 9, 22, 194, 224

        elif dataset_names[dataset_index] == '040119_PtK1_S01_01_phase_2_DMSO_nd_02':
            # top, left, bottom, right = 251, 324, 603, 501
            left, top, right, bottom = 228, 322, 800, 608
            rotate_angle_clockwise = 35
            expand_rotate = 1

        elif dataset_names[dataset_index] == '040119_PtK1_S01_01_phase_2_DMSO_nd_01':
            # top, left, bottom, right = 426, 166, 562, 307
            left, top, right, bottom = 46, 260, 161, 358
            rotate_angle_clockwise = 45

        elif dataset_names[dataset_index] == '040119_PtK1_S01_01_phase_ROI2':
            # top, left, bottom, right = 160, 433, 507, 589
            # rotate_angle_clockwise = 15
            left, top, right, bottom = 27, 191, 167, 460

        elif dataset_names[dataset_index] == '040119_PtK1_S01_01_phase':
            # top, left, bottom, right = 10, 84, 238, 452
            left, top, right, bottom = 4, 33, 278, 227

        elif dataset_names[dataset_index] == 'Paxilin-HaloTMR-TIRF3':
            left, top, right, bottom = 0, 0, 769, -1
            img_folder = 'img_all_cropped'

        elif dataset_names[dataset_index] == 'Paxilin-HaloTMR-TIRF4':
            left, top, right, bottom = 688, 479, 1306, 1154
            rotate_angle_clockwise = -30
            expand_rotate = 1
            img_folder = 'img_all'

        elif dataset_names[dataset_index] == 'Paxilin-HaloTMR-TIRF5':
            # img_all
            left, top, right, bottom = 802, 284, 1286, 1124
            rotate_angle_clockwise = -15
            expand_rotate = 1
            img_folder = 'img_all'

        elif dataset_names[dataset_index] == 'Paxilin-HaloTMR-TIRF7':
            left, top, right, bottom = 0, 0, -1, -1
            rotate_angle_clockwise = 0
            img_folder = 'img_all_cropped'

        elif dataset_names[dataset_index] == 'Paxilin-HaloTMR-TIRF8':
            left, top, right, bottom = 382, 442, 1182, 1244
            rotate_angle_clockwise = 30
            expand_rotate = 1
            img_folder = 'img_all'

        img_path = f"../assets/{dataset_names[dataset_index]}/{img_folder}/"

        img_save_path = f"generated/{dataset_names[dataset_index]}/{img_folder}/"
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)

        crop_images(img_path, img_format, 'img', img_save_path, expand_rotate, left, top, right, bottom,
                    rotate_angle_clockwise)
        crop_images(model_predict_path, img_format, 'mask', model_save_path, expand_rotate, left, top, right, bottom,
                    rotate_angle_clockwise)


def crop_for_paxillin():
    # dataset_names = ['Paxilin-HaloTMR-TIRF1', 'Paxilin-HaloTMR-TIRF2', 'Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4',
    #                  'Paxilin-HaloTMR-TIRF5', 'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
    dataset_names = ['Paxilin-HaloTMR-TIRF1', 'Paxilin-HaloTMR-TIRF2']
    img_format = '.png'

    for dataset_index in range(len(dataset_names)):
        print('dataset_index:', dataset_index)
        img_path = f"../assets/{dataset_names[dataset_index]}/img/"
        mask_path = f"../assets/{dataset_names[dataset_index]}/mask/"

        img_save_path = f"generated/{dataset_names[dataset_index]}/img_cropped/"
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)

        # model1_save_path = f"generated/{dataset_names[dataset_index]}/mask_cropped/"
        # if not os.path.exists(model1_save_path):
        #     os.makedirs(model1_save_path)

        expand_rotate = 0
        rotate_angle_clockwise = 0
        if dataset_index == 0:
            expand_rotate = 1
            rotate_angle_clockwise = 45
            left, top, right, bottom = 284, 328, 1340, 1236

        elif dataset_index == 1:
            expand_rotate = 1
            rotate_angle_clockwise = 15
            left, top, right, bottom = 342, 266, 1000, 898

        elif dataset_index == 2:
            left, top, right, bottom = 0, 114, 902, 1023

        elif dataset_index == 3:
            left, top, right, bottom = 405, 0, 1343, 1023

        elif dataset_index == 4:
            left, top, right, bottom = 425, 0, 1343, 1023

        elif dataset_index == 5:
            left, top, right, bottom = 573, 0, 1343, 1023

        elif dataset_index == 6:
            left, top, right, bottom = 89, 0, 1343, 838

        elif dataset_index == 7:
            left, top, right, bottom = 0, 0, 1343, 742  # for training and evaluation

        crop_images(img_path, img_format, 'img', img_save_path, expand_rotate, left, top, right, bottom,
                    rotate_angle_clockwise)
        # crop_images(mask_path, img_format, 'mask', model1_save_path, expand_rotate, left, top, right, bottom,
        #             rotate_angle_clockwise)


if __name__ == "__main__":
    crop_for_windowing()
    # crop_for_paxillin()

