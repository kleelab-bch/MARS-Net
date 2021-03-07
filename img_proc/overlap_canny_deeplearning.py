'''
Author Junbong Jang
8/18/2020

1. extract edge from deep learning segmentation
2. overlap canny edge from raw image and result from 1.
3. fill the edge
4. overlay edge with raw image

'''

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob
import os

from PIL import Image
from scipy import ndimage


def auto_canny(image, canny_std_multiplier):
    img_mean = np.average(image)
    img_std = np.std(image)
    
    lower = max(0, img_mean - (canny_std_multiplier+1)*img_std)
    upper = max(0, img_mean - canny_std_multiplier*img_std)
    # print('auto_canny:', img_mean, img_std, lower, upper)
    edges = cv2.Canny(image, lower, upper, 3, L2gradient=True)
    
    return edges
    

def extract_edge(img, img_name, saved_edge_path, canny_std_multiplier):
    canny_edge = auto_canny(img, canny_std_multiplier)
    canny_edge = pad_border_image(canny_edge)
    im = Image.fromarray(canny_edge)
    im.save(saved_edge_path + '/' + img_name)


def extract_edges(img_root_path, img_list, saved_edge_path, canny_std_multiplier):
    # extract edge for each original image
    for img_index in range(len(img_list)):
        # Get an image and its name
        img_path = img_list[img_index]
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = crop_border_image(img)
        
        img_name = img_path[len(img_root_path):]
        print(img_name, img.shape)
        
        extract_edge(img, img_name, saved_edge_path, canny_std_multiplier)


def crop_border_image(img):
    return img[10:, 10:]
    

def pad_border_image(img):
    img = np.pad(img, ((10,0),(10, 0)), 'constant')  # ((top, bottom), (left, right))
    return img


def overlap_edges(dl_canny_edge_list, img_canny_edge_list, img_list, saved_dl_edge_path, saved_overlap_path):
    for img_index in range(len(dl_canny_edge_list)):
        # Get an image and its name
        dl_canny_edge = cv2.imread(dl_canny_edge_list[img_index], cv2.IMREAD_GRAYSCALE)
        img_canny_edge = cv2.imread(img_canny_edge_list[img_index], cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_list[img_index], cv2.IMREAD_GRAYSCALE)
        
        dl_canny_edge = crop_border_image(dl_canny_edge)
        img_canny_edge = crop_border_image(img_canny_edge)
        img = crop_border_image(img)
        
        img_path = dl_canny_edge_list[img_index]
        img_name = img_path[len(saved_dl_edge_path):]
        print(img_name, img.shape, dl_canny_edge.shape, img_canny_edge.shape)
        
        img3 = np.zeros((img.shape[0],img.shape[1],3), dtype=img.dtype)
        #img3[:,:,0] = img
        img3[:,:,1] = img
        #img3[:,:,2] = img
        
        img3[:,:,0] = img3[:,:,0] + dl_canny_edge
        img3[:,:,2] = img3[:,:,2] + img_canny_edge
        
        im = Image.fromarray(img3)
        im.save(saved_overlap_path + '/' + img_name)


def overlay_edge_over_img(img, canny_edge, save_path):
    # overlay with the original image
    colorful_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    colorful_canny_edge = cv2.cvtColor(canny_edge, cv2.COLOR_GRAY2RGB)
    colorful_canny_edge[:, :, 1:2] = 0
    
    overlaid_img = cv2.addWeighted(colorful_img, 1, colorful_canny_edge, 0.3, 0)
    if save_path != '':
        im = Image.fromarray(overlaid_img)
        im.save(save_path)
    return overlaid_img


if __name__ == "__main__":
    canny_std_multiplier = 1
    constants = UserParams('predict')

    for dataset_index in range(0, len(constants.dataset)):
        a_dataset = constants.dataset[dataset_index]
        img_root_path = constants.dataset_folder + a_dataset + constants.img_folder
        saved_edge_path = f'generated/{a_dataset}/edge/'
        saved_overlap_path = f'generated/{a_dataset}/overlap/'
        if not os.path.exists(saved_edge_path):
            os.makedirs(saved_edge_path)
        if not os.path.exists(saved_overlap_path):
            os.makedirs(saved_overlap_path)


        # -----------------------------------
        # get prediction images
        # dl_prediction_root_path = f'../models/results/predict_wholeframe_round1_VGG16_trained/{a_dataset}/34_0_{a_dataset}/'
        # dl_prediction_list = glob.glob(dl_prediction_root_path + '*' + '.png')
        # print('number of images: ', len(dl_prediction_list))
        # extract_edges(dl_prediction_root_path, dl_prediction_list, saved_edge_path, canny_std_multiplier)

        # -----------------------------------
        # get mask images
        mask_list = glob.glob(constants.mask_folder + '*' + '.png')
        extract_edges(mask_root_path, dl_prediction_list, saved_edge_path, canny_std_multiplier)


        dl_canny_edge_list = glob.glob(saved_edge_path + '*' + '.png')
        img_canny_root_path = f'../label_tool/generated_edge/{a_dataset}/'
        img_canny_edge_list = glob.glob(img_canny_root_path + '*' + '.png')
        img_list = glob.glob(img_root_path + '*' + '.png')


        print(len(dl_canny_edge_list))
        print(len(img_canny_edge_list))
        overlap_edges(dl_canny_edge_list, img_canny_edge_list, img_list, saved_edge_path, saved_overlap_path)

        # ------------------------------------
    
    
    
    
        