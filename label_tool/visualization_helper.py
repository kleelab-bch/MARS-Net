'''
Author: Junbong Jang
7/28/2020

This is for visualization of the results
'''

import cv2
from PIL import Image


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