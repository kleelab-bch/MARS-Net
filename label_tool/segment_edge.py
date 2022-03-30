'''
Author: Junbong Jang
7/28/2020

For segmenting edges to train deep learning segmentation model
'''    
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import glob
import os
import random

from PIL import Image
from scipy import ndimage

from cv2.ximgproc import guidedFilter
import bm3d

import user_params
from visualization_helper import *
from extract_edge import auto_canny

 
def segment_edge(added_edge, fill_coords):
    #threshold edge
    added_edge[added_edge <= 110] = 0
    # added_edge[(added_edge > 110)] = 255
    # for exceptional user defined area
    added_edge[(added_edge > 110) & (added_edge != 136)] = 255
    added_edge[added_edge == 136] = 127 # this user defined gray area will be black
    
    # max_coord = get_coord_biggest_area(added_edge)
    
    # segment background
    filled_edge = added_edge.copy()
    for fill_coord in fill_coords:
        filled_edge = cv2.floodFill(filled_edge, None, fill_coord, newVal=127, loDiff=0, upDiff=126, flags=4)
        filled_edge = filled_edge[1]
    
    # segment foreground, threshold also
    filled_edge[filled_edge == 0] = 255
    filled_edge[filled_edge == 127] = 0
    copied_filled_edge = filled_edge.copy()
    
    # Remove small regions using contour area
    # https://stackoverflow.com/questions/60033274/how-to-remove-small-object-in-image-with-python
    cnts = cv2.findContours(copied_filled_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    max_area = 0
    # find the biggest area
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area

        # cv2.drawContours(copied_filled_edge,[c], -1, 255, -1)  # fill black hole
    
    cnts = cv2.findContours(copied_filled_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # remove other areas except the biggest
    for c in cnts:
        area = cv2.contourArea(c)
        if area < max_area:
            contour_cmap = (0,0,0)
            cv2.drawContours(copied_filled_edge, [c], -1, contour_cmap, -1)
    
    
    # remove small things that are jutting out
    kernel = np.ones((3,3),np.uint8)
    segmented_edge = cv2.morphologyEx(copied_filled_edge, cv2.MORPH_OPEN, kernel)

    # fill hole
    
    return segmented_edge
    
    
def segment_edges(img, img_name, edge_path, segment_path, fill_coord, canny_std_multiplier):
    extracted_edge = cv2.imread(edge_path + '/' + img_name,cv2.IMREAD_GRAYSCALE)

    # segmentation
    segmented_edge = segment_edge(extracted_edge, fill_coord)
    im = Image.fromarray(segmented_edge)
    im.save(segment_path + '/' + img_name)
    
    # overlay edge on the image
    save_path = segment_path + '/overlay_' + img_name
    canny_edge = auto_canny(segmented_edge, canny_std_multiplier)
    overlaid_img = overlay_edge_over_img(img, canny_edge, save_path)


# ---------- Helper Functions -------------------------

def get_coord_biggest_area(added_edge):
    # Find a point that fills the most area, which is most likely a background
    max_filled_area = 0
    max_coord = (0,0)  # x,y
    for coord in getCoordList(added_edge.shape):
        filled_edge = cv2.floodFill(added_edge.copy(), None, seedPoint=coord, newVal=127, loDiff=0, upDiff=126, flags=4)
        filled_edge = filled_edge[1]
    
        cur_filled_area = np.sum(filled_edge[filled_edge==127])
        if max_filled_area < cur_filled_area:
            max_filled_area = cur_filled_area
            max_coord = coord
    return max_coord
    
    
def getCoordList(image_shape):
    random.seed(a=42, version=2)
    # generate random coordinate points within the image shape
    coord_list = []
    rows, cols = image_shape
    rows, cols = rows - 1, cols - 1
    if rows > cols:
        for y in range(rows):
            x = random.randint(0, cols)
            coord_list.append((x,y))
    else:
        for x in range(cols):
            y = random.randint(0, rows)
            coord_list.append((x,y))
            
    return coord_list
    
    
def get_area_fill_coords():
    print('How many background areas to fill?')
    area_num = int(input('Enter 1~infinity:'))  # cols
    if area_num > 0:
        print()
    else:
        print('incorrect area input, terminating the program')
        exit(0)

    x_limit = img.shape[1]-1
    y_limit = img.shape[0]-1
    fill_coords = []
    print('Please input coordinate to fill the area')
    print(f'x,y coordinate of Top Left is (0,0) and Bottom Right is ({x_limit},{y_limit})', end='\n')
    
    for area_i in range(area_num):
        print(f'coord for area {area_i}')
        x_coord = int(input(f'Enter X position 0~{x_limit}:'))  # cols
        y_coord = int(input(f'Enter Y position 0~{y_limit}:'))  # rows
        if x_coord >= 0 and y_coord >= 0 and x_coord <= x_limit and y_coord <= y_limit:
            fill_coords.append((x_coord, y_coord))
            print(end='\n\n')
        else:
            print('incorrect coordinate input, terminating the program')
            exit(0)
    return fill_coords


if __name__ == "__main__":
    if not os.path.exists(user_params.saved_segment_path):
        os.makedirs(user_params.saved_segment_path)
        
    # get all of dataset's images
    img_list = glob.glob(user_params.img_root_path + '*' + '.png')
    print('number of images: ', len(img_list))
    
    # segment using each extracted edge image
    for img_index in range(len(img_list)):
        # Get an image and its name
        img_path = img_list[img_index]
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img_name = img_path[len(user_params.img_root_path):]
        print(img_name, img.shape)

        if img_index == 0:
            fill_coords = get_area_fill_coords()
        segment_edges(img, img_name, user_params.saved_edge_path, user_params.saved_segment_path, fill_coords, user_params.canny_std_multiplier)

    
