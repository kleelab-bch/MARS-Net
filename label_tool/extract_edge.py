'''
Author: Junbong Jang
6/10/2020

Automatically get the edge from the images to aid human labelers
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

from cv2.ximgproc import guidedFilter
# from cv2_rolling_ball import subtract_background_rolling_ball
# import bm3d

import user_params
from visualization_helper import *
    
    
# ------------ Image Processing Algorithms -------------------------
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(image, canny_std_multiplier):
    img_mean = np.average(image)
    img_std = np.std(image)
    
    lower = max(0, img_mean - (canny_std_multiplier+1)*img_std)
    upper = max(0, img_mean - canny_std_multiplier*img_std)
    # print('auto_canny:', img_mean, img_std, lower, upper)
    edges = cv2.Canny(image, lower, upper, 3, L2gradient=True)
    
    return edges
        

def clean_edge(input_edge):
    cleaned_edge = input_edge.copy()
    # Filter using contour area and remove small noise
    # https://stackoverflow.com/questions/60033274/how-to-remove-small-object-in-image-with-python
    cnts = cv2.findContours(cleaned_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    counter = 0
    for c in cnts:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c,False)
        if perimeter < 100:
            counter = counter + 1
            contour_cmap = (0,0,0)
            cv2.drawContours(cleaned_edge, [c], -1, contour_cmap, -1)
        '''
        elif area > 0:
            contour_cmap = (255,255,255)
            cv2.drawContours(cleaned_edge, [c], -1, contour_cmap, -1)
        '''
    print('number of deleted small edge:', str(counter))
    return cleaned_edge


def denoise_image(img, denoise_option, denoise_kernel_size):
    if denoise_option=='bm3d':
        denoised_image = bm3d.bm3d(img, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.ALL_STAGES) # .
        denoised_image = np.around(denoised_image,0).astype(np.uint8)#convert float to uint8
    elif denoise_option=='bilat':
        denoised_image = cv2.bilateralFilter(img,denoise_kernel_size,75,75)
    
    elif denoise_option=='median':
        denoised_image = cv2.medianBlur(img,denoise_kernel_size)
    
    elif denoise_option=='guided':
        r = 2; # try r=2, 4, or 8
        eps = 0.4 * 0.4; # try eps=0.1^2, 0.2^2, 0.4^2
        eps *= 255 * 255; # Because the intensity range of our images is [0, 255]
        denoised_image = guidedFilter(img,img,r,eps)
    
    elif denoise_option=='gauss':
        denoised_image = cv2.blur(img, (denoise_kernel_size,denoise_kernel_size))
    elif denoise_option=='kuwahara':
        denoised_image = Kuwahara(img, 13)
        
    else:
        denoised_image = None
    
    return denoised_image


def extract_edge(img, denoise_option, canny_std_multiplier, denoise_kernel_size, debug_mode):
    denoised_image = denoise_image(img, denoise_option, denoise_kernel_size)
    
    # subtracted_img, background = subtract_background_rolling_ball(denoised_image.copy(), 30, light_background=False, use_paraboloid=True, do_presmooth=False)

    canny_edge = auto_canny(denoised_image, canny_std_multiplier)
    
    # connects fragmented line one pixel away
    kernel = np.ones((2,2),np.uint8)
    dilated_canny_edge = cv2.dilate(canny_edge,kernel,iterations = 1)
    eroded_canny_edge = cv2.erode(dilated_canny_edge,kernel,iterations = 1)
                
    cleaned_edge = clean_edge(eroded_canny_edge)
    
    # --------
    if debug_mode:
        im = Image.fromarray(background)
        im.save('background_{}.png'.format(denoise_option))
        im = Image.fromarray(denoised_image)
        im.save('denoised_image_{}.png'.format(denoise_option))
        im = Image.fromarray(canny_edge)
        im.save('canny_edge_{}.png'.format(denoise_option))
        im = Image.fromarray(cleaned_edge)
        im.save('cleaned_edge_{}.png'.format(denoise_option))
    return cleaned_edge


def extract_edges(img, img_name, edge_path, canny_std_multiplier, denoise_kernel_size):
    # get edge from the image
    #bilat_edge = extract_edge(img, denoise_option='bilat', canny_std_multiplier, denoise_kernel_size, debug_mode=False)
    #gauss_edge = extract_edge(img, denoise_option='gauss', canny_std_multiplier, denoise_kernel_size, debug_mode=False)
    guided_edge = extract_edge(img, 'guided', canny_std_multiplier, denoise_kernel_size, debug_mode=False)
    
    # added_edge = gauss_edge + bilat_edge + guided_edge
    added_edge = guided_edge
    im = Image.fromarray(added_edge)
    im.save(edge_path + '/' + img_name)
    
    # orig image and edge overlaid image side by side
    save_path = edge_path + '/overlay_' + img_name
    overlaid_img = overlay_edge_over_img(img, added_edge, save_path='')
    
    visualize_two_images_side_by_side(img, overlaid_img, save_path)
   

def visualize_two_images_side_by_side(img, overlaid_img, save_path):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(img, cmap = 'gray')
    axarr[0].axis('off')
    axarr[1].imshow(overlaid_img)
    axarr[1].axis('off')
    
    plt.tight_layout(pad=0.2) # remove the padding between images
    plt.savefig(save_path,  bbox_inches = 'tight', pad_inches = 0, dpi=240) # remove up and down paddings
    plt.close()
    
    
if __name__ == "__main__":
    
    if not os.path.exists(user_params.saved_edge_path):
        os.makedirs(user_params.saved_edge_path)
        
    # get all of dataset's images
    img_list = glob.glob(user_params.img_root_path + '*' + '.png')
    print('number of images: ', len(img_list))
    
    # extract edge for each original image
    for img_index in range(len(img_list)):
        # Get an image and its name
        img_path = img_list[img_index]
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img_name = img_path[len(user_params.img_root_path):]
        print(img_name, img.shape)
        
        extract_edges(img, img_name, user_params.saved_edge_path, user_params.canny_std_multiplier, user_params.denoise_kernel_size)
        
        
            

    


