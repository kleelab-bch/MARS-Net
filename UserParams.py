'''
    Author : Junbong Jang
    Created in 3/9/2020
    
    Class to define user constants for cropping, training, prediction (segmentation), and evaluation
'''

import os
from models.model_utils import get_available_gpu, find_param_after_string
# tensorflow import must come after os.environ gpu setting
os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # get_available_gpu()
import argparse
import numpy as np
import random


class UserParams:
    def __init__(self, mode):
        """
        mode must be crop, train, predict, or eval
        """
        # if mode not in ['crop', 'train', 'predict', 'eval']:
        #     print('UserParams: mode is not crop, train, predict, or eval')
        #     exit()

        np.random.seed(seed=42)
        random.seed(42)

        self.round_num = 1 # [1,1,1,1,1,1,1,1,1]  # [1,2]
        # ['unet', 'VGG16', 'VGG19', 'VGG16_dropout', 'VGG19_dropout', 'Res50V2', 'EFF_B7_no_preprocessing']
        # ['specialist_unet', 'generalist_unet', 'specialist_VGG19_dropout', 'generalist_VGG19_dropout']
        # ['VGG19_dropout_input64', 'VGG19_dropout_input80', 'VGG19_dropout_input96', 'VGG19_dropout', 'VGG19_dropout_input192', 'VGG19_dropout_input256_crop200']
        # ['unet', 'VGG16_no_pretrain', 'VGG19_no_pretrain', 'VGG16', 'VGG19', 'VGG16_batchnorm', 'VGG19_batchnorm', 'VGG16_dropout', 'VGG19_dropout']
        # ['mDia_raw_unet', 'mDia_raw_VGG19_dropout']
        # ['paxillin_TIRF_normalize_cropped_unet_patience_10', 'paxillin_TIRF_normalize_cropped_VGG19_dropout_patience_10']
        self.strategy_type = 'FNA_VGG19_MTL_cls1_reg0_aut0_seg0.75_input256' # 'FNA_VGG19_MTL_auto_reg_aut_input256_patience_10' # 'FNA_VGG19_MTL_cls1_reg0_aut0_seg0.75_input256' # 'FNA_VGG19_MTL_auto_reg_aut_input256_patience_10' #'unet_imagenet_pretrained_marsnet' # 'FNA_CV_VGG19_MTL_auto_input256' # 'FNA_CV_VGG19_MTL_auto_input256' # 'unet_imagenet_pretrained_marsnet' # 'unet_encoder_classifier' # 'FNA_CV_VGG19_MTL_auto_reg_aut_input256' # 'spheroid_test_VGG19_marsnet' # 'FNA_VGG19_classifier_input256' #'FNA_CV_VGG19_classifier_binary_input256' # 'spheroid_VGG19_freeze' #'single_micro_small_unet' # 'cryptic_VGG19D_temporal_context_residual' # 'single_micro_VGG19D_temporal_context_residual' # 'cryptic_VGG19D_temporal_distributed_v2' # 'organoid_VGG19_dropout_crop_even' # 'cryptic_VGG19_dropout_mm_patience_10'
        self.self_training_type = None
        self.dataset_folders = '../assets/'
        self.img_format = '.png'
        self.crop_split_constant = 3
        self.REPEAT_MAX = 0

        if self.self_training_type is not None:
            self.strategy_type = self.strategy_type + '_' + self.self_training_type

        # dataset are ordered as follows E, D, C, B, A
        #self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
        #self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']  # you can choose these names
        #self.frame_list = [1,2,6,10,22,34]
        self.img_folder = '/img/'
        self.mask_folder = '/mask_fixed/'

        print('UserParams Status: ', mode, self.round_num, self.strategy_type)
        # ------------------------------------------------------

        if mode in ['crop','train']:
            if self.round_num == 1:
                self.crop_split_constant = 1
                self.img_folder = '/img/'

                if 'multi_micro' in str(self.strategy_type):
                    # if mode == 'train':  # don't crop since I manually move cropped files, commented in 3/15/2021
                    self.dataset_folders = ['../assets/','../assets/','../assets/','../assets/','../assets/',
                                           '../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/',
                                           '../assets/','../assets/','../assets/','../assets/','../assets/','../assets/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/',
                                        '/raw/','/raw/','/raw/','/raw/','/raw/',
                                        '/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/']
                    self.mask_folders = ['/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/']

                    self.frame_list = [2]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase', '1121-1', '1121-3', '1121-4', '1121-5', '1121-6',
                                          'Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                           'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    self.model_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
                    self.REPEAT_MAX = 1

                elif 'single_micro' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/','../assets/','../assets/','../assets/','../assets/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/']

                    self.frame_list = [10]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.REPEAT_MAX = 1

                elif 'spheroid_test' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/',
                                        '/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/',
                                        '/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/','/mask/']

                    self.frame_list = [1]
                    self.dataset_names = ['test','1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                          '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                          '20', '21', '22', '23']
                    self.model_names = ['train','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                        'Q', 'R', 'S', 'T', 'U', 'V', 'W']
                    self.REPEAT_MAX = 1

                elif 'spheroid' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/',
                                        '/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/',
                                        '/img/','/img/','/img/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/']

                    self.frame_list = [1]
                    self.dataset_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                          '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                          '20', '21', '22', '23']
                    self.model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                        'Q', 'R', 'S', 'T', 'U', 'V', 'W']
                    self.REPEAT_MAX = 1

                elif 'FNA_CV' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/FNA/','../assets/FNA/','../assets/FNA/','../assets/FNA/','../assets/FNA/','../assets/FNA/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask_processed/','/mask_processed/','/mask_processed/','/mask_processed/','/mask_processed/','/mask_processed/']

                    self.frame_list = [2]
                    self.dataset_names = ['FNA_valid_fold0', 'FNA_valid_fold1', 'FNA_valid_fold2', 'FNA_valid_fold3', 'FNA_valid_fold4', 'FNA_valid_fold5']
                    self.model_names = ['A','B','C','D','E','F']
                    self.REPEAT_MAX = 1

                elif 'FNA' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/FNA/','../assets/FNA/']
                    self.img_folders = ['/img/','/img/']
                    self.mask_folders = ['/mask_processed/','/mask_processed/']

                    self.frame_list = [2]
                    self.dataset_names = ['FNA_test', 'FNA_train_val']
                    self.model_names = ['training', '']
                    self.REPEAT_MAX = 1

                elif 'organoid_' in str(self.strategy_type):
                    self.dataset_folders = ['../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',]
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/']

                    self.frame_list = [2]
                    self.dataset_names = ['Lu-17_48h_z_ex1', 'Lu-17_48h_z_ex2', 'Lu-17_48h_z_ex4',
                                          'Lu-24_48h_z_ex1','Lu-24_48h_z_ex2','Lu-24_48h_z_ex3','Lu-24_48h_z_ex4',
                                          'Lu-24_724_z_ex1', 'Lu-24_724_z_ex2', 'Lu-24_724_z_ex4',
                                          'Lu-28_724_z_ex2', 'Lu-28_724_z_ex4']
                    self.model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
                    self.REPEAT_MAX = 1

                elif "mDia" in str(self.strategy_type):
                    self.dataset_folders = '../assets/mDia_chauncey/'
                    self.frame_list = [1,2,6,10,22,34]
                    self.dataset_names = ['1121-1', '1121-3', '1121-4', '1121-5', '1121-6']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.mask_folder = '/mask/'
                    self.REPEAT_MAX = 1
                    if "mDia_blurred" in str(self.strategy_type):
                        self.img_folder = '/blurred/'
                    if "mDia_denoised" in str(self.strategy_type):
                        self.img_folder = '/img_denoised/'
                    elif "mDia_raw" in str(self.strategy_type):
                        self.img_folder = '/raw/'

                elif "paxillin_TIRF" in str(self.strategy_type):
                    self.frame_list = [1,2,6,10,22]
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                    'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    self.model_names = ['ABCDE', 'ABCDF', 'ABCEF', 'ABDEF', 'ACDEF', 'BCDEF']

                    self.dataset_folders = ['../assets/', '../assets/', '../assets/', '../assets/', '../assets/', '../assets/']
                    self.img_folders = ['/img/', '/img/', '/img/', '/img/', '/img/', '/img/']
                    self.mask_folders = ['/mask/', '/mask/', '/mask/', '/mask/', '/mask/', '/mask/']

                    if "_cropped" in str(self.strategy_type):
                        self.img_folders = ['/img_cropped/', '/img_cropped/', '/img_cropped/',
                                            '/img_cropped/', '/img_cropped/', '/img_cropped/']
                        self.mask_folders = ['/mask_cropped/', '/mask_cropped/', '/mask_cropped/',
                                             '/mask_cropped/', '/mask_cropped/', '/mask_cropped/']

                    self.REPEAT_MAX = 1

                elif "paxillin_WF" in str(self.strategy_type):
                    self.dataset_folders = '../../assets/'
                    self.frame_list = [1,2,6,10,22]
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF3-WF_TMR_M', 'Paxilin-HaloTMR-TIRF4-WF_TMR_M', 'Paxilin-HaloTMR-TIRF5-WF_TMR_M',
                                    'Paxilin-HaloTMR-TIRF6-WF_TMR_M', 'Paxilin-HaloTMR-TIRF7-WF_TMR_M', 'Paxilin-HaloTMR-TIRF8-WF_TMR_M']
                    self.model_names = ['ABCDE', 'ABCDF', 'ABCEF', 'ABDEF', 'ACDEF', 'BCDEF']
                    self.mask_folder = '/mask/'
                    self.REPEAT_MAX = 1

                elif "mask_denoising" == str(self.strategy_type):
                    '''
                    self.dataset_folders = '../../../HeeJune/Image_Data/PtK1_CyD/Windowing/111017/'
                    self.img_folder = '/SegmentationPackage/masks/masks_for_channel_1/'
                    self.img_format = '.tif'
                    '''
                    self.dataset_folders = '../../../HeeJune/Image_Data/PtK1_CyD/PtK1_CyD_Pair/Windowing_whole_3px'
                    self.img_folder = '/images_png/masks_for_channel_1_png/'
                    self.frame_list = [2]
                    self.dataset_names = self.get_datasets_in_path(self.dataset_folders, ['etc', 'Video_mask_thumbnails_051920', 'Video_thumbnails'])
                    self.model_names = [self.strategy_type]
                    self.mask_folder = '/SegmentationPackage/refined_masks/refined_masks_for_channel_1/'

                elif "cryptic_combined" in str(self.strategy_type):

                    self.dataset_folders = ['../assets/','../assets/','../assets/','../assets/','../assets/',
                                           '../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/',
                                           '../assets/','../assets/','../assets/','../assets/','../assets/','../assets/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/',
                                        '/raw/','/raw/','/raw/','/raw/','/raw/',
                                        '/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/',
                                        '/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/']
                    self.frame_list = [10]

                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase', '1121-1', '1121-3', '1121-4', '1121-5', '1121-6',
                                          'Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                          'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8',
                                          '101018_part_E', '101018_part_D', '101018_part_C', '101018_part_B', '101018_part_A']
                    self.model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']

                    self.REPEAT_MAX = 1

                elif "cryptic_all" in str(self.strategy_type):
                    self.img_folder = '/img/'
                    self.mask_folder = '/mask/'
                    self.dataset_folders = '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/'
                    self.frame_list = [10]
                    self.dataset_names = ['101018_all']
                    self.model_names = ['A']
                    self.REPEAT_MAX = 1

                elif "cryptic_" in str(self.strategy_type):
                    self.img_folders = ['/img_all/','/img_all/','/img_all/','/img_all/','/img_all/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/']
                    self.dataset_folders = ['../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/']

                    self.frame_list = [2]
                    self.dataset_names = ['101018_part_E', '101018_part_D', '101018_part_C', '101018_part_B', '101018_part_A']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']

                    self.REPEAT_MAX = 1

                elif self.strategy_type == 6:
                    self.img_folder = '/img_unlabelled/'
                    self.frame_list = [2]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD']

                elif self.strategy_type == 'dice' or self.strategy_type == 'l2' or self.strategy_type == "VGG16_input256" or self.strategy_type == "VGG16_dac_input256" or self.strategy_type == "VGG16_spp_input256":
                    self.img_folder = '/img/'
                    self.frame_list = [2]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']

                elif self.strategy_type == 'VGG19_dropout_input256_crop200':
                    self.img_folder = '/img/'
                    self.frame_list = [1, 2, 6, 10, 22]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.REPEAT_MAX = 1

                elif 'VGG16' in self.strategy_type or 'VGG19' in self.strategy_type or self.strategy_type == 'movie3' or 'unet' in self.strategy_type or \
                     self.strategy_type == 'Res50V2' or self.strategy_type == 'Dense201' or self.strategy_type == 'deeplabv3' or self.strategy_type == 'EFF_B7' or \
                     self.strategy_type == 'EFF_B7_no_preprocessing' or self.strategy_type == 'InceptionResV2':

                    self.dataset_folders = ['../assets/','../assets/','../assets/','../assets/','../assets/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/']

                    self.frame_list = [1,2,6,10,22,34]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.REPEAT_MAX = 1

            elif self.round_num == 2:  # self-training
                self.img_folder = '/img_all/'

                if "paxillin_TIRF_normalize_2.5" == str(self.strategy_type):
                    self.frame_list = [200]
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                    'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    self.model_names = ['ABCDE']
                    self.REPEAT_MAX = 1
                    self.crop_split_constant = 3

                elif "cryptic" in str(self.strategy_type):
                    self.dataset_names = ['101018_part_E', '101018_part_D', '101018_part_C', '101018_part_B', '101018_part_A']
                    self.frame_list = [110]
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.dataset_folders = '../../../HeeJune/Segmentation_Image/Cryptic Lamellipodia/CellMask-05152014-Control-1/'

                elif self.self_training_type == 2:
                    self.img_folder = '/img/'
                    self.crop_split_constant = 1
                    self.dataset_names = ['040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2', '040119_PtK1_S01_01_phase']
                    self.frame_list = [34]
                    self.model_names = ['ABCD']

                elif self.self_training_type == 2.5:
                    self.crop_split_constant = 1
                    self.frame_list = [1,2,6,10,22,34]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']

                elif self.self_training_type in [5,6]:
                    self.frame_list = [2,6,10,22,34,70]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD']

                elif self.self_training_type in [1,3]:
                    self.crop_split_constant = 5
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03']
                    self.frame_list = [200]
                    self.model_names = ['ABCD']

                elif self.strategy_type == 'movie3' or self.strategy_type == 'movie3_loss' or self.strategy_type == 'movie3_proc':
                    self.frame_list = [34]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']  # TODO: reduce model names to ABCD?

        
        elif mode in ['predict']:
            self.img_folder = '/img/'
            if self.round_num == 1:

                if 'generalist' in str(self.strategy_type) or 'specialist' in str(self.strategy_type):
                    if 'feature_extractor_big_orig' in str(self.strategy_type):
                        self.dataset_folders = '../assets/test_feature_extractor_big_orig/'
                    elif 'feature_extractor_big' in str(self.strategy_type):
                        self.dataset_folders = '../assets/test_feature_extractor_big/'
                    elif 'feature_extractor_small' in str(self.strategy_type):
                        self.dataset_folders = '../assets/test_feature_extractor_small/'
                    else:
                        self.dataset_folders = '../assets/test_generalist/'
                    self.dataset_folders = ['../assets/', '../assets/', '../assets/', '../assets/', '../assets/',
                                            '../assets/mDia_chauncey/', '../assets/mDia_chauncey/',
                                            '../assets/mDia_chauncey/', '../assets/mDia_chauncey/',
                                            '../assets/mDia_chauncey/',
                                            '../assets/', '../assets/', '../assets/', '../assets/', '../assets/',
                                            '../assets/']
                    self.img_folders = ['/img_all/', '/img_all/', '/img_all/', '/img_all/', '/img_all/',
                                        '/raw/', '/raw/', '/raw/', '/raw/', '/raw/',
                                        '/img_all_cropped/', '/img_all_cropped/', '/img_all_cropped/', '/img_all_cropped/',
                                        '/img_all_cropped/', '/img_all_cropped/']
                    self.mask_folders = ['/mask_fixed/', '/mask_fixed/', '/mask_fixed/', '/mask_fixed/', '/mask_fixed/',
                                         '/mask/', '/mask/', '/mask/', '/mask/', '/mask/',
                                         '/mask_cropped/', '/mask_cropped/', '/mask_cropped/', '/mask_cropped/',
                                         '/mask_cropped/', '/mask_cropped/']
                    self.frame_list = [2]

                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase', '1121-1', '1121-3', '1121-4', '1121-5', '1121-6',
                                          'Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                          'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    self.model_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
                    if 'one_generalist' in str(self.strategy_type):
                        self.model_names = ['All']
                    self.REPEAT_MAX = 1

                elif 'multi_micro' in str(self.strategy_type):
                    # if mode == 'train':  # don't crop since I manually move cropped files, commented in 3/15/2021
                    self.dataset_folders = ['../assets/','../assets/','../assets/','../assets/','../assets/',
                                           '../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/',
                                           '../assets/','../assets/','../assets/','../assets/','../assets/','../assets/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/',
                                        '/raw/','/raw/','/raw/','/raw/','/raw/',
                                        '/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/']
                    self.mask_folders = ['/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/']

                    self.frame_list = [6,10]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase', '1121-1', '1121-3', '1121-4', '1121-5', '1121-6',
                                          'Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                           'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    self.model_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P']
                    self.REPEAT_MAX = 1

                elif 'single_micro' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/','../assets/','../assets/','../assets/','../assets/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/']

                    self.frame_list = [10]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.REPEAT_MAX = 1

                elif 'spheroid_test' in str(self.strategy_type):

                    # self.dataset_folders = ['../assets/Spheroid/test/','../assets/Spheroid/test/','../assets/Spheroid/test/',
                    #                         '../assets/Spheroid/test/','../assets/Spheroid/test/','../assets/Spheroid/test/',
                    #                         '../assets/Spheroid/test/','../assets/Spheroid/test/','../assets/Spheroid/test/',
                    #                         '../assets/Spheroid/test/','../assets/Spheroid/test/','../assets/Spheroid/test/']
                    self.dataset_folders = ['../assets/Spheroid/test_2048/','../assets/Spheroid/test_2048/','../assets/Spheroid/test_2048/',
                                            '../assets/Spheroid/test_2048/','../assets/Spheroid/test_2048/','../assets/Spheroid/test_2048/',
                                            '../assets/Spheroid/test_2048/','../assets/Spheroid/test_2048/','../assets/Spheroid/test_2048/',
                                            '../assets/Spheroid/test_2048/','../assets/Spheroid/test_2048/','../assets/Spheroid/test_2048/']
                    self.img_folders = ['/','/','/','/','/','/',
                                        '/','/','/','/','/','/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/','/mask/']

                    self.frame_list = [1]
                    self.dataset_names = ['Cdc42_5uM', 'Cdc42_10uM', 'DMSO', 'DMSO_2', 'FAK', 'FAK_2',
                                          'Rac_5uM', 'Rac_10uM', 'Rac_20uM', 'Rho_5uM', 'Rho_10uM', 'Rho_20uM']

                    self.model_names = ['train', 'train', 'train', 'train', 'train', 'train',
                                        'train', 'train', 'train', 'train', 'train', 'train']
                    self.REPEAT_MAX = 1

                elif 'spheroid' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/',
                                        '/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/',
                                        '/img/','/img/','/img/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/']

                    self.frame_list = [1]
                    self.dataset_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                          '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                          '20', '21', '22', '23']
                    self.model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                        'Q', 'R', 'S', 'T', 'U', 'V', 'W']
                    self.REPEAT_MAX = 1

                elif 'FNA_CV' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/FNA/','../assets/FNA/','../assets/FNA/','../assets/FNA/','../assets/FNA/','../assets/FNA/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask_processed/','/mask_processed/','/mask_processed/','/mask_processed/','/mask_processed/','/mask_processed/']

                    self.frame_list = [2]
                    self.dataset_names = ['FNA_valid_fold0', 'FNA_valid_fold1', 'FNA_valid_fold2', 'FNA_valid_fold3', 'FNA_valid_fold4', 'FNA_valid_fold5']
                    self.model_names = ['A','B','C','D','E','F']
                    self.REPEAT_MAX = 1

                elif 'FNA' in str(self.strategy_type):
                    self.dataset_folders = ['../assets/FNA/', '../assets/FNA/']
                    self.img_folders = ['/img/', '/img/']
                    self.mask_folders = ['/mask_processed/', '/mask_processed/']

                    self.frame_list = [2]
                    self.dataset_names = ['FNA_test', 'FNA_train_val']
                    self.model_names = ['training', '']
                    self.REPEAT_MAX = 1

                elif 'organoid_' in str(self.strategy_type):
                    self.dataset_folders = ['../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',
                                            '../../Organoid/generated/segmentation_train/','../../Organoid/generated/segmentation_train/',]
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/']

                    self.frame_list = [2]
                    self.dataset_names = ['Lu-17_48h_z_ex1', 'Lu-17_48h_z_ex2', 'Lu-17_48h_z_ex4',
                                          'Lu-24_48h_z_ex1','Lu-24_48h_z_ex2','Lu-24_48h_z_ex3','Lu-24_48h_z_ex4',
                                          'Lu-24_724_z_ex1', 'Lu-24_724_z_ex2', 'Lu-24_724_z_ex4',
                                          'Lu-28_724_z_ex2', 'Lu-28_724_z_ex4']
                    self.model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
                    self.REPEAT_MAX = 1

                elif "cryptic_all" in str(self.strategy_type):
                    self.img_folder = '/img/'
                    self.mask_folder = '/mask/'
                    self.dataset_folders = '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/'
                    self.frame_list = [10]
                    self.dataset_names = ['101018_all']
                    self.model_names = ['A']
                    self.REPEAT_MAX = 1

                elif "cryptic_combined" in str(self.strategy_type):

                    self.dataset_folders = ['../assets/','../assets/','../assets/','../assets/','../assets/',
                                           '../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/',
                                           '../assets/','../assets/','../assets/','../assets/','../assets/','../assets/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/',
                                        '/raw/','/raw/','/raw/','/raw/','/raw/',
                                        '/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/',
                                        '/img/','/img/','/img/','/img/','/img/']
                    self.mask_folders = ['/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/','/mask_fixed/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/']
                    self.frame_list = [10]

                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase', '1121-1', '1121-3', '1121-4', '1121-5', '1121-6',
                                          'Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                          'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8',
                                          '101018_part_E', '101018_part_D', '101018_part_C', '101018_part_B', '101018_part_A']
                    self.model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']

                    self.REPEAT_MAX = 1

                elif "cryptic" in str(self.strategy_type):
                    self.img_folders = ['/img_all/','/img_all/','/img_all/','/img_all/','/img_all/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/']
                    self.dataset_folders = ['../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/',
                                            '../assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/']
                    self.frame_list = [2]
                    self.dataset_names = ['101018_part_E', '101018_part_D', '101018_part_C', '101018_part_B', '101018_part_A']
                    self.model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.REPEAT_MAX = 1

                elif "mDia" in str(self.strategy_type):
                    self.dataset_folders = '../assets/mDia_chauncey/'
                    self.frame_list = [1,2,6,10,22,34]
                    self.dataset_names = ['1121-1', '1121-3', '1121-4', '1121-5', '1121-6']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.mask_folder = '/mask/'
                    self.REPEAT_MAX = 1
                    if "mDia_blurred" in str(self.strategy_type):
                        self.img_folder = '/blurred/'
                    if "mDia_denoised" in str(self.strategy_type):
                        self.img_folder = '/img_denoised/'
                    elif "mDia_raw" in str(self.strategy_type):
                        self.img_folder = '/raw/'

                elif "paxillin_TIRF" in str(self.strategy_type):
                    self.frame_list = [1,2,6,10,22]
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                    'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    self.model_names = ['ABCDE', 'ABCDF', 'ABCEF', 'ABDEF', 'ACDEF', 'BCDEF']

                    # self.frame_list = [22]
                    # self.dataset_names = ['Paxilin-HaloTMR-TIRF5']
                    # self.model_names = ['ABCEF']
                    # self.REPEAT_MAX = 1
                    # self.img_folder = '/img_all/'

                    if "_cropped" in str(self.strategy_type):
                        self.img_folder = '/img_cropped/'
                        self.mask_folder = '/mask_cropped/'

                elif "paxillin_WF" in str(self.strategy_type):
                    self.dataset_folders = '../../assets/'
                    self.frame_list = [1, 2, 3, 4]
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF3-WF_TMR_M', 'Paxilin-HaloTMR-TIRF4-WF_TMR_M',
                                    'Paxilin-HaloTMR-TIRF5-WF_TMR_M',
                                    'Paxilin-HaloTMR-TIRF6-WF_TMR_M', 'Paxilin-HaloTMR-TIRF7-WF_TMR_M',
                                    'Paxilin-HaloTMR-TIRF8-WF_TMR_M']
                    self.model_names = ['ABCDE', 'ABCDF', 'ABCEF', 'ABDEF', 'ACDEF', 'BCDEF']
                    self.REPEAT_MAX = 1

                elif self.strategy_type == 'VGG19_dropout_input256_crop200':
                    self.frame_list = [1, 2, 6, 10, 22]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.REPEAT_MAX = 1

                elif self.strategy_type == "VGG16_input256" or self.strategy_type == "VGG16_dac_input256" or self.strategy_type == "VGG16_spp_input256" or self.strategy_type == 'dice' or self.strategy_type == 'l2':
                    self.frame_list = [2]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.REPEAT_MAX = 1

                elif 'VGG16' in self.strategy_type or 'VGG19' in self.strategy_type or self.strategy_type == 'movie3' or 'unet' in self.strategy_type or \
                     self.strategy_type == 'Res50V2' or self.strategy_type == 'Dense201' or self.strategy_type == 'deeplabv3' or self.strategy_type == 'EFF_B7' or \
                     self.strategy_type == 'EFF_B7_no_preprocessing' or self.strategy_type == 'InceptionResV2':

                    self.dataset_folders = ['../assets/', '../assets/', '../assets/', '../assets/', '../assets/']
                    self.img_folders = ['/img/', '/img/', '/img/', '/img/', '/img/']
                    self.mask_folders = ['/mask_fixed/', '/mask_fixed/', '/mask_fixed/', '/mask_fixed/', '/mask_fixed/']

                    self.frame_list = [1,2,6,10,22,34]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.REPEAT_MAX = 1

                    # self.img_folders = ['/raw/','/raw/','/raw/','/raw/','/raw/',
                    #                     '/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/','/img_cropped/']
                    # self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/',
                    #                      '/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/','/mask_cropped/']
                    # self.frame_list = [10]
                    #
                    # self.dataset_names = ['1121-1', '1121-3', '1121-4', '1121-5', '1121-6',
                    #                       'Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                    #                       'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    # self.model_names = ['ABCD','ABCD','ABCD','ABCD','ABCD',
                    #                     'ABCD','ABCD','ABCD','ABCD','ABCD','ABCD']
                    # self.REPEAT_MAX = 1
                    # self.dataset_folders = ['../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/','../assets/mDia_chauncey/',
                    #                        '../assets/','../assets/','../assets/','../assets/','../assets/','../assets/']

                elif self.strategy_type == 2:
                    self.dataset_names = ['040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.frame_list = [2]
                    self.model_names = ['ABCD']
                    self.REPEAT_MAX = 1

                elif self.strategy_type == 2.5:
                    self.img_folder = '/img_all/'
                    self.dataset_names = ['040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.frame_list = [1]
                    self.model_names = ['ABCD']
                    self.REPEAT_MAX = 1

            elif self.round_num == 2:
                if self.strategy_type == 3:
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03']
                    self.frame_list = [200]
                    self.model_names = ['ABCD']
                if self.strategy_type == 2:
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03']
                    self.frame_list = [34]
                    self.model_names = ['ABCD']
                elif self.strategy_type == 2.5 or self.strategy_type == 5:
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2', '040119_PtK1_S01_01_phase']
                    self.frame_list = [34]
                    self.model_names = ['ABCD']
                elif self.strategy_type == 'movie3' or self.strategy_type == 'movie3_loss' or self.strategy_type == 'movie3_proc':
                    self.frame_list = [34]
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2','040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                elif "paxillin_TIRF_normalize_2.5" in str(self.strategy_type):
                    self.frame_list = [200]
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF3']
                    self.model_names = ['ABCDE']
                    self.REPEAT_MAX = 1


        elif mode in ['eval']:
            self.predict_path_list = ['../../models/results/predict_wholeframe_round{}_{}/'.format(a_round_num, a_strategy_type) for a_strategy_type, a_round_num in zip(self.strategy_type, self.round_num)]

        if mode in ['crop', 'train', 'predict']:
            if not self.model_names or not self.dataset_names or not self.frame_list:
                print('UserParams: Wrong Mode or settings @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                exit()

            assert (len(self.dataset_folders) == len(self.dataset_names))
            assert (len(self.model_names) == len(self.dataset_names))
            assert (len(self.mask_folders) == len(self.dataset_names))
            assert (len(self.img_folders) == len(self.dataset_names))


    # ------------- helper functions -----------
    def get_args(self):
        parser = argparse.ArgumentParser()

        crop_mode = 'random'
        if 'crop_even' in str(self.strategy_type) or 'temporal_' in str(self.strategy_type) or 'FNA_' in str(self.strategy_type):
            crop_mode = 'even'

        if 'patience_10' in str(self.strategy_type):
            patience = 10
            epochs = 100
        else:
            patience = 3
            epochs = 100

        if 'input1024' in str(self.strategy_type):
            crop_mode = crop_mode + '_input1024'
            input_size = 1024
            crop_patches = 12
            crop_batch_size = 8
            train_batch_size = 16
        elif 'input512' in str(self.strategy_type):
            crop_mode = crop_mode + '_input512'
            input_size = 512
            crop_patches = 24
            crop_batch_size = 16
            train_batch_size = 16
        elif "input256_crop200" in str(self.strategy_type):
            input_size = 256
            crop_patches = 200
            crop_batch_size = 128
        elif "input256" in str(self.strategy_type):
            crop_mode = crop_mode + '_input256'
            input_size = 256
            crop_patches = 50
            crop_batch_size = 32
            train_batch_size = 32
        elif "input192" in str(self.strategy_type):
            crop_mode = crop_mode + '_input192'
            input_size = 192
            crop_patches = 200
            crop_batch_size = 128
            train_batch_size = 64
        elif "_3D" in str(self.strategy_type):
            input_size = 128
            crop_patches = 200
            crop_batch_size = 128
            train_batch_size = 16
        elif "input96" in str(self.strategy_type):
            input_size = 96
            crop_patches = 200
            crop_batch_size = 128
            train_batch_size = 64
        elif "input80" in str(self.strategy_type):
            input_size = 80
            crop_patches = 200
            crop_batch_size = 128
            train_batch_size = 64

        elif "input64" in str(self.strategy_type):
            input_size = 64
            crop_patches = 200
            crop_batch_size = 128
            train_batch_size = 64
        elif "temporal" in str(self.strategy_type):
            input_size = 128
            crop_patches = 200
            crop_batch_size = 128
            train_batch_size = 32
        else:
            input_size = 128
            crop_patches = 200
            crop_batch_size = 128
            train_batch_size = 64

        output_size = input_size - 60

        parser.add_argument("--input_size", type=int, default=input_size)
        parser.add_argument("--input_depth", type=int, default=16)
        parser.add_argument("--output_size", type=int, default=output_size)
        parser.add_argument("--crop_mode", type=str, default=crop_mode)
        parser.add_argument("--crop_patches", type=int, default=crop_patches)
        parser.add_argument("--crop_batch_size", type=int, default=crop_batch_size) # how many images to augment at once
        parser.add_argument("--cropped_boundary", type = int, default = 30)
        parser.add_argument("--train_batch_size", type = int, default = train_batch_size)
        parser.add_argument("--epochs", type = int, default = epochs)
        parser.add_argument("--validation_split", type = int, default = 0.2)
        parser.add_argument("--patience", type = int, default = patience)

        args = parser.parse_args("")
        return args
    
    
    def get_datasets_in_path(self, dataset_folders, exclude_names=['etc']):
        folder_names_in_path = [name for name in os.listdir(dataset_folders) if os.path.isdir(os.path.join(dataset_folders, name)) ]
        # Exclude some folders
        for exclude_name in exclude_names:
            if exclude_name in folder_names_in_path:
                folder_names_in_path.remove(exclude_name)
        
        return folder_names_in_path
        
        
    def get_mask_folder(self, model_name, dataset_name, frame, repeat_index):
        # for cropping images in self_training
        mask_folder_root = '../models/results/predict_wholeframe_round{}_{}/{}_{}/'.format(
            (self.round_num-1), self.strategy_type, model_name, dataset_name)

        if self.round_num >= 2:
            if "cryptic" in str(self.strategy_type) or self.self_training_type == 2:
                self.mask_folder = mask_folder_root + 'frame{}_repeat{}/'.format(frame, repeat_index)

            elif "paxillin_TIRF_normalize_2.5" == self.strategy_type:
                self.mask_folder = mask_folder_root + 'processed_frame{}_repeat{}/'.format(frame, repeat_index)

            elif self.self_training_type in [1,2.5,3,5,6] or self.strategy_type == 'movie3_proc' or \
                    self.strategy_type == 'movie3' or self.strategy_type == 'movie3_loss':
                self.mask_folder = mask_folder_root + 'processed_frame{}_repeat{}/'.format(frame, repeat_index)

        print('get_mask_folder: ' + self.mask_folder)
        return self.mask_folder


    def get_crop_path(self, model_name, dataset_name, frame, split_index, repeat_index):
        # specify the folder in which to save cropped result
        root_path = '../crop/crop_results/crop_round{}_{}/'.format(self.round_num, self.strategy_type)

        if self.round_num == 1:
            if 'cryptic' in str(self.strategy_type):
                if '_combined' in str(self.strategy_type):
                    root_path = '../crop/crop_results/crop_round1_VGG16/'
                elif '_heq' in str(self.strategy_type):
                    root_path = '../crop/crop_results/crop_round1_cryptic_heq/'
                else:
                    root_path = '../crop/crop_results/crop_round1_cryptic/'
            elif '_input256_crop200' in str(self.strategy_type):
                root_path = '../crop/crop_results/crop_round1_input256_output196_crop200/'
            elif '_input256' in str(self.strategy_type):
                root_path = '../crop/crop_results/crop_round1_input256_output196/'
            elif self.strategy_type == 2 or self.strategy_type == 'VGG16_tf' or self.strategy_type == 'VGG16_no_pretrain' or \
                    self.strategy_type == 'VGG16_batchnorm' or self.strategy_type == 'VGG16_dropout' or self.strategy_type == 'VGG16_instancenorm' or \
                    self.strategy_type == 'VGG19_dropout' or self.strategy_type == 'specialist_VGG19_dropout' or \
                    self.strategy_type == 'VGG19' or self.strategy_type == 'VGG19_no_pretrain' or \
                    self.strategy_type == 'VGG19_batchnorm' or self.strategy_type == 'VGG19_dropout_batchnorm' or \
                    self.strategy_type == 'Res50V2' or self.strategy_type == 'Dense201' or \
                    self.strategy_type == 'deeplabv3' or self.strategy_type == 'EFF_B7' or self.strategy_type == 'InceptionResV2' or \
                    'imagenet_pretrained' in self.strategy_type or \
                    'unet' in self.strategy_type or self.strategy_type == 'dice' or self.strategy_type == 'l2':
                root_path = '../crop/crop_results/crop_round1_VGG16/'

            elif 'paxillin_TIRF_normalize_cropped' in self.strategy_type:
                root_path = '../crop/crop_results/crop_round1_paxillin_TIRF_normalize_cropped/'
            elif 'paxillin_TIRF_cropped' in self.strategy_type:
                root_path = '../crop/crop_results/crop_round1_paxillin_TIRF_cropped/'
            elif 'paxillin_TIRF_normalize' in self.strategy_type:
                root_path = '../crop/crop_results/crop_round1_paxillin_TIRF_normalize/'

            elif 'mDia_blurred_test' in self.strategy_type:
                root_path = '../crop/crop_results/crop_round1_mDia_blurred_test/'
            elif 'mDia_raw_test' in self.strategy_type:
                root_path = '../crop/crop_results/crop_round1_mDia_raw_test/'
            elif 'mDia_denoised' in self.strategy_type:
                root_path = '../crop/crop_results/crop_round1_mDia_denoised/'
            elif 'mDia_raw' in self.strategy_type:
                root_path = '../crop/crop_results/crop_round1_mDia_raw/'
            elif 'mDia' in self.strategy_type:
                root_path = '../crop/crop_results/crop_round1_mDia/'

            elif 'generalist' in str(self.strategy_type):
                root_path = '../crop/crop_results/crop_round1_generalist/'

        elif self.round_num == 2:
            # e.g) put crops inside the folder ABCD_round1, BCDE_round1
            root_path = root_path + model_name + '/'

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        save_suffix = '{}_frame{}_split{}_repeat{}_train_mask.npz'.format(dataset_name, frame, split_index, repeat_index)
        print('get_crop_path: ' + root_path + save_suffix)

        return root_path, save_suffix
        

    def get_pretrained_weights_path(self, frame, model_name):
        weights_path = ''  # default is VGG16 imagenet pretrained weight
        if "VGG16_custom" == str(self.strategy_type) or "VGG16_batchnorm" == str(self.strategy_type) or "VGG16_instancenorm" == str(self.strategy_type)\
                or "VGG16_dropout" == str(self.strategy_type):
            weights_path = '../vgg16_weights.h5'
        elif 'VGG19_dropout_sm' in str(self.strategy_type):
            weights_path = '../models/results/model_round1_VGG19_dropout/model_frame34_ABCD_repeat0.hdf5'
        elif 'VGG19_dropout_mm' in str(self.strategy_type):
            weights_path = '../models/results/model_round1_generalist_VGG19_dropout/model_frame2_A_repeat0.hdf5'
        elif 'VGG19_imagenet_pretrained' in str(self.strategy_type):
            weights_path = '../models/results/model_round1_VGG19_imagenet_classifier/model_frame0_A_repeat0.hdf5'
        elif 'unet_imagenet_pretrained' in str(self.strategy_type):
            weights_path = '../models/results/model_round1_unet_encoder_classifier/model_frame0_A_repeat0.hdf5'
        elif "mask_denoising" == str(self.strategy_type):
            weights_path = 'results/model_round{}_{}/model'.format(self.round_num, self.strategy_type)+str(frame)+'_' + model_name +'.hdf5'

        print('@@@@@ get_pretrained_weights_path: ' + weights_path)
        return weights_path
        
        
    def get_trained_weights_path(self, frame, model_name, repeat_index, root_path = ''):
        weights_path = 'results/model_round{}_{}/model_'.format(self.round_num, self.strategy_type)

        # special case for student learning
        if self.round_num == 1:
            if self.strategy_type == 2 or self.strategy_type == 2.5 or self.strategy_type == 3 or self.strategy_type == 5:
                weights_path = 'results/model_round1_VGG16/model_'
            elif self.strategy_type == 'paxillin_TIRF_normalize_2.5':
                weights_path = 'results/model_round1_paxillin_TIRF_normalize/model_'

            if 'feature_extractor' in str(self.strategy_type):
                if 'one_generalist' in str(self.strategy_type):
                    if 'generalist_unet' in str(self.strategy_type):
                        weights_path = 'results/model_round1_one_generalist_unet_feature_extractor/model_'
                    elif 'generalist_VGG19_dropout' in str(self.strategy_type):
                        weights_path = 'results/model_round1_one_generalist_VGG19_dropout_feature_extractor/model_'

                elif 'generalist_VGG19_dropout' in str(self.strategy_type):
                    weights_path = 'results/model_round1_generalist_VGG19_dropout/model_'
                elif 'generalist_unet' in str(self.strategy_type):
                    weights_path = 'results/model_round1_generalist_unet_feature_extractor/model_'
                elif 'specialist_VGG19_dropout' in str(self.strategy_type):
                    weights_path = 'results/model_round1_specialist_VGG19_dropout/model_'
                elif 'specialist_unet' in str(self.strategy_type):
                    weights_path = 'results/model_round1_specialist_unet/model_'

        save_suffix = 'frame{}_{}_repeat{}.hdf5'.format(frame, model_name, repeat_index)
        weights_path = root_path + weights_path + save_suffix
        print('get_trained_weights_path: ' + weights_path)
        return weights_path


    def get_datasets_for_model(self, model_name):
        '''
         This function assumes all model_names and datasets are provided.
         Also, dataset and model names are arranged in reverse order as follows:
         dataset = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                         '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                         '040119_PtK1_S01_01_phase']
         model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']

         Extract each letter from string 'ABCD' and map it to the dataset A --> dataset, B --> dataset2
         Then, get a list of datasets used to build the model ABCD
        '''
        unique_model_letters = list(set(''.join(self.model_names)))
        unique_model_letters = sorted(unique_model_letters, key=str.lower)  # e.g) 'A', 'B', 'C', 'D', 'E'
        assert(len(unique_model_letters) == len(self.model_names))

        dataset_list = []
        for a_letter in model_name:
            dataset_index = len(self.dataset_names) - unique_model_letters.index(a_letter) - 1
            dataset_list.append(self.dataset_names[dataset_index])
        assert (len(dataset_list)>2)
        return dataset_list

    # --------------- For evaluation -------------

    def update_eval_config(self, predict_path_index):
        predict_path = self.predict_path_list[predict_path_index]
        # print('update_eval_config', predict_path_index, predict_path)

        self.img_folder = '/img/'

        if self.round_num[predict_path_index] == 1:
            if 2 in self.round_num:
                if "predict_wholeframe_round1_paxillin_TIRF" in str(predict_path):
                    self.dataset_folders = '../../assets/'
                    self.mask_folder = '/mask/'
                    self.model_names = ['ABCDE']
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF3']
                    self.frame_list = [2]
                    self.REPEAT_MAX = 1

                elif "predict_wholeframe_round1_VGG" in str(predict_path) or 'predict_wholeframe_round1_unet' in str(
                        predict_path):
                    self.dataset_folders = '../../assets/'
                    self.mask_folder = '/mask_fixed/'
                    self.model_names = ['ABCD']
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03']
                    self.frame_list = [2]
                    self.REPEAT_MAX = 1

            else:
                if 'generalist' in str(self.strategy_type) or 'specialist' in str(self.strategy_type):
                    self.dataset_folders = '../../assets/test_generalist/'
                    self.img_folder = '/img/'
                    self.mask_folder = '/mask/'
                    self.frame_list = [2]

                    # self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03',
                    #                       '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                    #                       '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                    #                       '040119_PtK1_S01_01_phase', '1121-1', '1121-3', '1121-4', '1121-5', '1121-6']
                    # self.model_names = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J']

                    # self.dataset_names = ['Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                    #                       'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    # self.model_names = ['K', 'L', 'M', 'N', 'O', 'P']

                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                          '040119_PtK1_S01_01_phase', '1121-1', '1121-3', '1121-4', '1121-5', '1121-6',
                                          'Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                          'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    self.model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
                    if "one_generalist" in str(predict_path):
                        self.dataset_folders = '../../assets/test_one_generalist/'
                        self.model_names = ['All']
                    self.REPEAT_MAX = 1

                elif 'spheroid' in str(predict_path):
                    self.dataset_folders = ['../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/','../assets/Spheroid/',
                                            '../assets/Spheroid/','../assets/Spheroid/']
                    self.img_folders = ['/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/',
                                        '/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/','/img/',
                                        '/img/','/img/','/img/']
                    self.mask_folders = ['/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/','/mask/',
                                         '/mask/','/mask/','/mask/']

                    self.frame_list = [1]
                    self.dataset_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                          '11', '12', '13', '14', '15', '16', '17', '18', '19',
                                          '20', '21', '22', '23']
                    self.model_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                        'Q', 'R', 'S', 'T', 'U', 'V', 'W']
                    self.REPEAT_MAX = 1

                elif "predict_wholeframe_round1_mDia" in str(predict_path):
                    self.dataset_folders = '../../assets/mDia_chauncey/'
                    self.mask_folder = '/mask/'
                    self.model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.dataset_names = ['1121-1', '1121-3', '1121-4', '1121-5', '1121-6']
                    self.frame_list = [1,2,6,10,22,34]
                    self.REPEAT_MAX = 1

                elif "predict_wholeframe_round1_paxillin_TIRF" in str(predict_path):
                    self.dataset_folders = '../../assets/'
                    self.mask_folder = '/mask/'
                    self.model_names = ['ABCDE', 'ABCDF', 'ABCEF', 'ABDEF', 'ACDEF', 'BCDEF']
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF3', 'Paxilin-HaloTMR-TIRF4', 'Paxilin-HaloTMR-TIRF5',
                                    'Paxilin-HaloTMR-TIRF6', 'Paxilin-HaloTMR-TIRF7', 'Paxilin-HaloTMR-TIRF8']
                    self.frame_list = [1,2,6,10,22]
                    self.REPEAT_MAX = 1

                    if "_cropped" in str(self.strategy_type):
                        self.img_folder = '/img_cropped/'
                        self.mask_folder = '/mask_cropped/'

                elif "predict_wholeframe_round1_paxillin_WF" in str(predict_path):
                    self.dataset_folders = '../../assets/'
                    self.mask_folder = '/mask/'
                    self.model_names = ['ABCDE', 'ABCDF', 'ABCEF', 'ABDEF', 'ACDEF', 'BCDEF']
                    self.dataset_names = ['Paxilin-HaloTMR-TIRF3-WF_TMR_M', 'Paxilin-HaloTMR-TIRF4-WF_TMR_M',
                                    'Paxilin-HaloTMR-TIRF5-WF_TMR_M',
                                    'Paxilin-HaloTMR-TIRF6-WF_TMR_M', 'Paxilin-HaloTMR-TIRF7-WF_TMR_M',
                                    'Paxilin-HaloTMR-TIRF8-WF_TMR_M']
                    self.frame_list = [1, 2, 6, 10, 22]
                    self.REPEAT_MAX = 1

                elif "predict_wholeframe_round1_single_micro" in str(predict_path):  # phase contrast images
                    self.dataset_folders = '../../assets/'
                    self.mask_folder = '/mask_fixed/'
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                    '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                    '040119_PtK1_S01_01_phase']
                    self.model_names = ['A', 'B', 'C', 'D', 'E']
                    self.frame_list = [10]
                    self.REPEAT_MAX = 3

                else:  # phase contrast images
                    self.dataset_folders = '../../assets/'
                    self.mask_folder = '/mask_fixed/'
                    self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02',
                                    '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2',
                                    '040119_PtK1_S01_01_phase']
                    self.model_names = ['ABCD', 'ABCE', 'ABDE', 'ACDE', 'BCDE']
                    self.frame_list = [10]
                    self.REPEAT_MAX = 3

        elif self.round_num[predict_path_index] == 2:
            if "predict_wholeframe_round2_paxillin_TIRF" in str(predict_path):
                self.model_names = ['ABCDE']
                self.dataset_names = ['Paxilin-HaloTMR-TIRF3']
                self.frame_list = [200]
                self.REPEAT_MAX = 1

            elif "predict_wholeframe_round2_VGG" in str(predict_path):
                self.model_names = ['ABCD']
                self.dataset_names = ['040119_PtK1_S01_01_phase_3_DMSO_nd_03']
                self.frame_list = [34]
                self.REPEAT_MAX = 1


    def get_eval_folder_name(self):
        folder_name = ''
        for index, (a_round_num, a_strategy_type) in enumerate(zip(self.round_num, self.strategy_type)):
            folder_name = folder_name + 'round' + str(a_round_num) + '_' + a_strategy_type
            if index < len(self.round_num) - 1:
                folder_name = folder_name + '_'

        return folder_name

    def get_save_prediction_path(self, dataset_name, model_name, frame, repeat_index):

        root_prediciton_path = "results/predict_wholeframe_round{}_{}/".format(self.round_num,
                                                                               self.strategy_type)

        if self.self_training_type is None:
            save_path = root_prediciton_path + '{}/frame{}_{}_repeat{}/'.format(dataset_name, str(frame), model_name,
                                                                     str(repeat_index))
        else:
            save_path = root_prediciton_path + '{}_{}/frame{}_repeat{}/'.format(model_name, dataset_name, str(frame),
                                                                     str(repeat_index))
        print('save_path:', save_path)
        if os.path.isdir(save_path) == 0:
            os.makedirs(save_path)

        return save_path
