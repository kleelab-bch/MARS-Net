"""
Created on 3/23/2020
@author: Junbong Jang
"""
import os
import shutil
import sys
sys.path.append('..')
import constants

#dataset = ['040119_PtK1_S01_01_phase',  '040119_PtK1_S01_01_phase_ROI2', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_3_DMSO_nd_03']
dataset = constants.dataset
if __name__ == '__main__':
    for index in range(0, len(dataset), 1):
        img_all_path = '../assets/' + dataset[index] + '/img_all/'
        img_labelled_path = '../assets/' + dataset[index] + '/img/'
        img_unlabelled_path = '../assets/' + dataset[index] + '/img_unlabelled/'
        
        img_all = os.listdir(img_all_path)
        img_labelled = os.listdir(img_labelled_path)
        img_unlabelled = list(set(img_all) - set(img_labelled))
        print(dataset[index])
        print(len(img_all))
        print(len(img_labelled))
        print(len(img_unlabelled))
        for a_img_unlabelled in img_unlabelled:
            dest = shutil.copyfile(img_all_path + a_img_unlabelled, img_unlabelled_path + a_img_unlabelled) 