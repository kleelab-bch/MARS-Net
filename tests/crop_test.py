'''
Author Junbong Jang
Date 4/16/2021

To test functions in crop/crop.py and crop/crop_helpers.py
Reference https://theaisummer.com/unit-test-deep-learning/
'''
import unittest

import numpy as np
import os, cv2
import glob
from tqdm import tqdm

import sys
sys.path.append('..')
from UserParams import UserParams
from crop.crop_helpers import data_generate
from sklearn.model_selection import train_test_split
import utils_test
sys.path.append('../models')
from train_mars import get_training_dataset


class CropTest(unittest.TestCase):

    def test_sample_loc_at_diff_repeat(self):
        constants = UserParams('crop')
        # constants conditioned on strategy_type = single_micro_VGG19D_dropout
        # self.dataset_folders = ['../assets/', '../assets/', '../assets/', '../assets/', '../assets/']
        # self.img_folders = ['/img/', '/img/', '/img/', '/img/', '/img/']
        # self.mask_folders = ['/mask_fixed/', '/mask_fixed/', '/mask_fixed/', '/mask_fixed/', '/mask_fixed/']
        #
        # self.frame_list = [2]
        # self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
        #                       '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
        #                       '040119_PtK1_S01_01_phase']
        # self.model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']
        # self.REPEAT_MAX = 2

        args = constants.get_crop_args()
        print(args)
        repeat_max = 5

        dataset_folder = constants.dataset_folders[0]
        img_folder = constants.img_folders[0]
        mask_folder = constants.mask_folders[0]
        dataset_name = constants.dataset_names[0]
        round_num = constants.round_num
        input_size = args.input_size
        output_size = args.output_size
        img_format = constants.img_format
        crop_mode = args.crop_mode
        crop_patches = args.crop_patches
        print('@@-@@', dataset_folder, img_folder, mask_folder, dataset_name)

        total_frames = 40
        for i in range(0,total_frames,20):
            for repeat_index in range(repeat_max):
                print(i, repeat_index)
                np.random.seed(seed=42)
                data_generator = data_generate(dataset_name, input_size, output_size, repeat_index, round_num,
                                               img_format,
                                               crop_mode, crop_patches, dataset_folder, img_folder, mask_folder)
                images, masks, framenames = data_generator.read_img_mask()

                self.assertEqual(len(framenames), data_generator.total_frames)

                # crop rand
                edge_ratio = 0.6
                image, mask = images[i], masks[i]

                edge_number = int(data_generator.rand_crop_num * edge_ratio)
                back_number = data_generator.rand_crop_num - edge_number

                edge = cv2.Canny(mask, 100, 200)

                loc_p, sample_p = data_generator.sample_loc(edge, edge_number, on_edge=True)
                loc_n, sample_n = data_generator.sample_loc(edge, back_number, on_edge=False)

                # print(loc_p[0].shape, loc_p[1].shape, sample_p, sample_p[0])
                # print(loc_p[0][sample_p[0]], loc_p[1][sample_p[0]])
                # print(loc_n[0].shape, loc_n[1].shape, sample_n, sample_n[0])
                # print(loc_n[0][sample_n[0]], loc_n[1][sample_n[0]])
                # print(sample_p)
                # print(sample_n)

                if repeat_index == 0:
                    old_loc_p, old_sample_p = loc_p, sample_p
                    old_loc_n, old_sample_n = loc_n, sample_n
                else:
                    np.testing.assert_array_equal(old_loc_p, loc_p)
                    np.testing.assert_array_equal(old_loc_n, loc_n)

                    self.assertTrue(np.any(np.not_equal(old_sample_p, sample_p)))
                    self.assertTrue(np.any(np.not_equal(old_sample_n, sample_n)))

                    old_loc_p, old_sample_p = loc_p, sample_p
                    old_loc_n, old_sample_n = loc_n, sample_n

    # for crop_augment_split.py crops
    def test_show_crops(self):
        constants = UserParams('crop')
        model_index = 0
        frame = 2
        repeat_index = 0
        training_dataset = get_training_dataset(constants, model_index, frame, repeat_index)
        comb_train = training_dataset['arr_0']
        comb_mask = training_dataset['arr_1']

        img_path = constants.dataset_folders[model_index] + '/' + constants.dataset_names[model_index] + constants.img_folder

        utils_test.show_cropped_image(comb_train, comb_mask, img_path, constants.img_format, constants.strategy_type,
                           f'../tests/generated/crop_test/round{constants.round_num}_{constants.strategy_type}/{constants.dataset_names[model_index]}_frame{frame}_repeat{repeat_index}/')


    def test_crop_framenames(self):
        dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                              '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                              '040119_PtK1_S01_01_phase']
        for dataset_name in dataset_names:
            frame = 2
            repeat_index = 0
            index_path = '../crop/crop_results/crop_round1_single_micro_VGG19_dropout/{}_frame{}_repeat{}_train_index.npy'.format(dataset_name, frame, repeat_index)
            name_path = '../crop/crop_results/crop_round1_single_micro_VGG19_dropout/{}_frame{}_repeat{}_train_names.npy'.format(dataset_name, frame, repeat_index)
            index_data = np.load(index_path, allow_pickle=True, encoding="bytes")
            name_data = np.load(name_path, allow_pickle=True, encoding="bytes")
            print(dataset_name, name_data[index_data])


    def test_split_val_consistent_after_repeat(self):
        print('test_split_val_consistent_after_repeat')
        train_0, test_0, train_1, test_1 = train_test_split([0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], test_size=0.2, random_state=42)
        train_2, test_2, train_3, test_3 = train_test_split([0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], test_size=0.2, random_state=42)
        print(train_0, test_0, train_1, test_1)
        self.assertEqual(train_0, train_2)
        self.assertEqual(test_0, test_2)
        self.assertEqual(train_1, train_3)
        self.assertEqual(test_1, test_3)


if __name__ == '__main__':
    unittest.main()