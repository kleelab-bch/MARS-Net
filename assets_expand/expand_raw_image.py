# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:56:18 2019

@author: Chauncey Wang
"""
import os, cv2
import numpy as np
from data_generator import prediction_data_generate

import sys
sys.path.append('..')
import constants

#dataset = ['040119_PtK1_S01_01_phase', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_03']
dataset = constants.dataset
if __name__ == '__main__':
    #Expand the size of data.
    folder = '../assets/'
    saved_folder = '../assets_expand/'
    for index in range(0, len(dataset), 1):
        img_path = folder + dataset[index] + '/img/'
        msk_path = folder + dataset[index] + '/mask/'
        train = prediction_data_generate(img_path, msk_path, 1, 128, 68, 10)
        imgs, mask, namespace, image_cols, image_rows = train.get_whole_frames()
        
        #Save the image back.
        img_path = saved_folder + dataset[index] + '/img/'
        msk_path = saved_folder + dataset[index] + '/mask/'
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        if not os.path.exists(msk_path):
            os.makedirs(msk_path)
        for saved_index in range(0, len(namespace), 1):
            img_name = namespace[saved_index]
            cv2.imwrite(img_path + img_name, imgs[saved_index])
            #filemask = 'mask' + img_name[3:] 
            filemask = img_name
            cv2.imwrite(msk_path + filemask, mask[saved_index])
            

    