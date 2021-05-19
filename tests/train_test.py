'''
Author Junbong Jang
Date 4/19/2021

To test functions in models/train.py and models/train_data_generator.py

'''
import unittest
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2

import sys
sys.path.append('../models')
import utils_test
from train_data_generator import get_training_dataset_names, calc_augmentation_factor, getAugmentedImages, GetImageMask
import debug_utils

class TrainTest(unittest.TestCase):
    def test_calc_augmentation_factor_low(self):
        model_name = 'ABCD'
        frame = 2
        repeat_index = 0
        save_path = '../models/results/history_round1_single_micro_VGG19_dropout/'
        batch_size = 128

        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = get_training_dataset_names(model_name,
                                                                                                            frame,
                                                                                                            repeat_index,
                                                                                                            save_path)

        self.assertEqual(calc_augmentation_factor(x_train_filenames, batch_size, 'train_val'), 202)
        self.assertEqual(calc_augmentation_factor(x_train_filenames, batch_size, 'train'), 160)
        self.assertEqual(calc_augmentation_factor(x_val_filenames, batch_size, 'valid'), 40)

    def test_calc_augmentation_factor_high(self):
        model_name = 'A'
        frame = 34
        repeat_index = 0
        save_path = '../models/results/history_round1_single_micro_VGG19_dropout/'
        batch_size = 128

        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = get_training_dataset_names(model_name,
                                                                                                            frame,
                                                                                                            repeat_index,
                                                                                                            save_path)


        self.assertEqual(calc_augmentation_factor(x_train_filenames, batch_size, 'train_val'), 42)
        self.assertEqual(calc_augmentation_factor(x_val_filenames, batch_size, 'valid'), 0)

    def test_get_augmented_images(self):
        model_name = 'ABCD'
        frame = 2
        repeat_index = 0
        save_path = '../models/results/history_round1_single_micro_VGG19_dropout/'
        batch_size = 64


        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = get_training_dataset_names(model_name,
                                                                                                            frame,
                                                                                                            repeat_index,
                                                                                                            save_path)

        # ------------- Check that mean, std and distribution are within expectation -----------

        # train_x, train_y = getAugmentedImages(x_train_filenames, y_train_filenames, batch_size, 'standardize', 'train')
        valid_x, valid_y = getAugmentedImages(x_val_filenames, y_val_filenames, batch_size, 'standardize', 'valid')

        # print('train_x:', np.ptp(train_x), np.mean(train_x), np.std(train_x), train_x.shape)
        # print('train_y:', np.ptp(train_y), np.mean(train_y), np.std(train_y), train_y.shape)
        print('valid_x:', np.ptp(valid_x), np.mean(valid_x), np.std(valid_x), valid_x.shape, valid_x.dtype)
        print('valid_y:', np.ptp(valid_y), np.mean(valid_y), np.std(valid_y), valid_y.shape, valid_y.dtype)
        # self.assertAlmostEqual(np.mean(train_x), 0, places=1)
        # self.assertAlmostEqual(np.std(train_x), 1, places=1)
        # self.assertAlmostEqual(np.ptp(train_y), 1, places=3)
        # self.assertAlmostEqual(np.mean(valid_x), 0, places=1)
        # self.assertAlmostEqual(np.std(valid_x), 1, places=1)
        # self.assertAlmostEqual(np.ptp(valid_y), 1, places=3)
        #
        # # ----------- Check that all images are not 0 matrix ----------
        # print('@@@@@')
        # for x in tqdm(train_x):
        #     self.assertNotEqual(np.sum(x), 0)
        # for x in tqdm(valid_x):
        #     self.assertNotEqual(np.sum(x), 0)

        # ----------------- Save Images
        test_augmented_path = '../tests/generated/train_test/'
        index_list = [x for x in range(100)] + [x for x in range(valid_x.shape[0]-1, valid_x.shape[0]-201, -1)]

        batch_x, batch_y = GetImageMask(x_val_filenames, y_val_filenames)
        overlay_edge_array = valid_x.copy() * np.std(batch_x) + np.mean(batch_x)
        for index in tqdm(index_list):
            a_mask = valid_y[index, 0, :, :] * 255
            a_mask = a_mask.astype('uint8')

            utils_test.save_overlay_img(overlay_edge_array[index,0,30:98,30:98], cv2.Canny(a_mask, 100, 200), test_augmented_path, index)


    # def test_get_cropped_images_mars(self):
    #     dataset_name = ''
    #     dataset_folder = ''
    #     img_format = '.png'
    #     frame = 2
    #     repeat_index = 0
    #     save_path = '../models/results/history_round1_single_micro_VGG19_dropout/'
    #     batch_size = 64
    #
    #     training_dataset = get_training_dataset(constants, model_index, frame, repeat_index)
    #     comb_train = training_dataset['arr_0']
    #     comb_mask = training_dataset['arr_1']
    #
    #     # ------------ process dataset ----------------
    #     comb_train, comb_mask = unison_shuffled_copies(comb_train, comb_mask)
    #     img_path = dataset_folder + '/' + dataset_name + constants.img_folder
    #     show_cropped_image(comb_train, comb_mask, img_path, img_format, strategy_type,
    #                        f'results/debugger/round1_{strategy_type}/{dataset_name}_frame{frame}_repeat{repeat_index}/')
    #
    #
    # def test_open_history_data(self):
    #     strategy_type = 'single_micro_VGG19_dropout'
    #     model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']
    #     frame = 2
    #     for model_name in model_names:
    #         root_path = '../models/results/history_round1_{}/history_frame{}_{}_repeat0.npy'.format(strategy_type,
    #                                                                                                 frame, model_name)
    #         data = np.load(root_path, allow_pickle=True, encoding="bytes")
    #         data_dict = data.ravel()[0]
    #         print(data_dict)
    #         # print('Total training in hours: ', sum(data_dict['times']) / 3600)

    def check_loaded_weights(self):
        #TODO
        print(debug_utils.check_loaded_weights(weights_path))

if __name__ == '__main__':
    unittest.main()
