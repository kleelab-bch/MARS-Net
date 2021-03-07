# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:32:03 2019

@author: Chauncey
"""
import numpy as np
import os, cv2
import glob

dataset = ['0601-4', '0601-5', '0601-1', '0321-2', '0328-5', '0328-6', '0328-3', '0601-1-half']


def read_msk(msk_f):
   msk = cv2.imread(msk_f, cv2.IMREAD_GRAYSCALE)
   #msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
   msk[msk>0] = 255
   edg = cv2.Canny(msk,100,200)
   edg[edg>0] = 255
   return msk, edg

if __name__ == "__main__":
    for i in range(len(dataset)):
        #load the images from source
        image_data = dataset[i]
        m_path = image_data + '/mask/'
        mask_list = glob.glob(os.path.join(m_path, '*.png'))
        #save the path 
        saved_path = image_data + '/mask_bin/'
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        total_number = len(mask_list)
        #binarize the image
        for j in range(total_number):
            msks, edgs = read_msk(mask_list[j])
            saved_name = mask_list[j]
            saved_name = saved_name[len(m_path):]
            cv2.imwrite(saved_path + saved_name, msks)