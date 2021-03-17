'''
Author Junbong Jang
Creation Date 3/9/2021

processing functions for images and masks before training, or prediction
'''

import numpy as np
import cv2


def to3channel(imgs):
    imgs = imgs[:,np.newaxis,:,:]
    imgs_p = np.repeat(imgs, 3, axis=1)
    imgs_p = imgs_p.astype('float32')

    return imgs_p


def preprocess_input(imgs):
    mean = np.mean(imgs)
    std = np.std(imgs)

    imgs_p = to3channel(imgs)
    imgs_p -= mean
    imgs_p /= std

    return imgs_p


def preprocess_output(imgs):
    imgs = imgs[:,np.newaxis,:,:]
    imgs_p = imgs.astype('float32')
    imgs_p /= 255.  # scale masks to [0, 1]

    return imgs_p


def normalize_input(imgs):
    imgs_p = to3channel(imgs)
    imgs_p /= 255.  # scale image to [0, 1]
    return imgs_p


def normalize_clip_input(imgs):
    std, mean = np.std(imgs), np.mean(imgs)

    imgs_p = to3channel(imgs)

    max_val = mean + 3 * std
    min_val = mean - 3 * std
    if min_val < 0:
        min_val = 0
    if max_val > 255:
        max_val = 255
    np.clip(imgs_p, min_val, max_val, out=imgs_p)
    imgs_p = (imgs_p - min_val) / (max_val - min_val)

    return imgs_p


def heq_norm_input(imgs):
    imgs_heq = np.ndarray(imgs.shape, dtype=np.uint8)
    for img_index in range(imgs_heq.shape[0]):
        imgs_heq[img_index,0] = cv2.equalizeHist(imgs[img_index,0])

    imgs_p = to3channel(imgs_heq)
    imgs_p /= 255.  # scale image to [0, 1]

    return imgs_p


