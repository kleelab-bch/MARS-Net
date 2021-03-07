'''
Author: Junbong Jang
6/10/2020

Automatically get the edge from the images before inputting into deep learning segmentation model
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
from cv2_rolling_ball import subtract_background_rolling_ball
import bm3d


def auto_canny_with_dillation(image, sigma=0.3, dillation_size=32):
    v = np.average(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    print('auto_canny_with_dillation:',v, lower, upper)
    edges = cv2.Canny(image, lower, upper, 3, L2gradient=True)
    
    kernel = np.ones((dillation_size,dillation_size),np.uint8)
    dilation = cv2.dilate(edges,kernel,iterations = 1)
    
    return dilation
    

def extract_edge_image_from_mask(img, mask, debug_mode):
    img = cv2.blur(img, (5,5))
    canny_edge = auto_canny(img)
    cropped_edge = canny_edge & auto_canny_with_dillation(mask, dillation_size=16)

    cleaned_edge = clean_edge(cropped_edge)
    
    if debug_mode:
        im = Image.fromarray(canny_edge)
        im.save('canny edge.png')
        im = Image.fromarray(auto_canny_with_dillation(mask, dillation_size=16))
        im.save('dillated mask edge.png')
        im = Image.fromarray(cropped_edge)
        im.save('cropped edge.png')
        im = Image.fromarray(cleaned_edge)
        im.save('cleaned_edge.png')
        
    return cleaned_edge
    

def sobel_image(img):
    # Output dtype = cv2.CV_64F. 
    # Then take its absolute and convert to cv2.CV_8U
    sobel = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    abs_sobel64f = np.absolute(sobel)
    sobel_x = np.uint8(abs_sobel64f)
    
    sobel = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    abs_sobel64f = np.absolute(sobel)
    sobel_y = np.uint8(abs_sobel64f)
    
    return sobel_x, sobel_y, sobel_x+sobel_y
    

def draw_comprison_images(img, mask):

    img, background = subtract_background_rolling_ball(img, 10, light_background=True, use_paraboloid=True, do_presmooth=False)

    blur_img = cv2.blur(img, (4,4))
    blur_mask = cv2.blur(mask, (4,4))
    # -----------------------
    blur_sobel_x, blur_sobel_y, blur_sobel_xy = sobel_image(blur_img)
    '''
    scharr = cv2.Scharr(img,cv2.CV_64F,dx=0,dy=1)
    abs_scharr = np.absolute(scharr)
    scharr_8u = np.uint8(abs_scharr)

    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    abs_laplacian = np.absolute(laplacian)
    lapacian_8u = np.uint8(abs_laplacian)
    '''
    blur_canny_auto = auto_canny(blur_img)
    #-----------------------
    mask_sobel_x, mask_sobel_y, mask_sobel_xy = sobel_image(mask)
    mask_canny_auto = auto_canny(mask)
    # ----------------------
    blur_mask_sobel_x, blur_mask_sobel_y, blur_mask_sobel_xy = sobel_image(blur_mask)
    blur_mask_canny_auto = auto_canny(blur_mask)
    # ----------------------
    
    edge = img & auto_canny_with_dillation(mask)
    
    plt.subplot(1,5,1),plt.imshow(img,cmap = 'gray')
    plt.title('image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,5,2),plt.imshow(mask,cmap = 'gray')
    plt.title('mask'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,5,3),plt.imshow(edge,cmap = 'gray')
    plt.title('Image Cropped'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,5,4),plt.imshow(auto_canny(edge),cmap = 'gray')
    plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,5,5),plt.imshow(auto_canny_with_dillation(mask, dillation_size=8),cmap = 'gray')
    plt.title('Canny Mask'), plt.xticks([]), plt.yticks([])
    
    plt.tight_layout(pad=1.0)
    plt.savefig('edge_image.png', dpi=300)
    plt.close() 
    
    blur_edge = cv2.blur(edge, (4,4))
    edge_sobel_x, edge_sobel_y, edge_sobel_xy = sobel_image(blur_edge)
    plt.subplot(1,5,1),plt.imshow(blur_edge,cmap = 'gray')
    plt.title('Blur Edge'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,5,2),plt.imshow(auto_canny(blur_edge),cmap = 'gray')
    plt.title('Canny'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,5,3),plt.imshow(edge_sobel_xy,cmap = 'gray')
    plt.title('Sobel X+Y'), plt.xticks([]), plt.yticks([])
    
    edge_sobel_xy_mean = np.mean(edge_sobel_xy[edge_sobel_xy!=0])
    edge_sobel_xy_std = np.std(edge_sobel_xy[edge_sobel_xy!=0])
    threshold_edge_sobel_xy = np.ndarray(edge_sobel_xy.shape)
    print(edge_sobel_xy_mean, edge_sobel_xy_std)
    for x_ind, columns in enumerate(edge_sobel_xy):
        for y_ind, column in enumerate(columns):
            if column > edge_sobel_xy_mean+edge_sobel_xy_std or column < edge_sobel_xy_mean:
                column = 0
            threshold_edge_sobel_xy[x_ind,y_ind] = column
    
    threshold_edge_sobel_xy = threshold_edge_sobel_xy.astype(np.uint8)
    new_edge = threshold_edge_sobel_xy & auto_canny_with_dillation(mask, dillation_size=8)
    
    plt.subplot(1,5,4),plt.imshow(threshold_edge_sobel_xy,cmap = 'gray')
    plt.title('Threshold'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,5,5),plt.imshow(new_edge,cmap = 'gray')
    plt.title('Cropped'), plt.xticks([]), plt.yticks([])
    
    plt.tight_layout(pad=1.0)
    plt.savefig('edge_image2.png', dpi=300)
    plt.close() 
    
    
def draw_image_histogram(img):
    ax = plt.hist(img.ravel(), bins = 256)
    plt.xlabel('pixel value')
    plt.ylabel('frequency')
    plt.title('Histogram of image')
    plt.savefig('histogram.png')
    
    
def threshold_image(img):
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
                
    titles = ['Original Image', 'Global Thresholding (v = 127)',
        'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]

    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.savefig('threshold.png')

    
if __name__ == "__main__":
    img = cv2.imread('040119_PtK1_S01_01_phase_3_DMSO_nd_03_001_kuwahara.png',cv2.IMREAD_GRAYSCALE)
    
    kuwahara_edge = extract_edge_image(img, denoise_option='kuwahara', debug_mode=True)
    bilat_edge = extract_edge_image(img, denoise_option='bilat', debug_mode=True)
    gauss_edge = extract_edge_image(img, denoise_option='gauss', debug_mode=True)
    guided_edge = extract_edge_image(img, denoise_option='guided', debug_mode=True)
    
    im = Image.fromarray(bilat_edge+gauss_edge+guided_edge)
    im.save('addition_edges.png')

'''
    mask = cv2.imread('../assets/040119_PtK1_S01_01_phase_3_DMSO_nd_03/mask/040119_PtK1_S01_01_phase_3_DMSO_nd_03_045.png',cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('../assets/040119_PtK1_S01_01_phase_3_DMSO_nd_03/img/040119_PtK1_S01_01_phase_3_DMSO_nd_03_045.png',cv2.IMREAD_GRAYSCALE)
    
    draw_comprison_images(img,mask)
'''