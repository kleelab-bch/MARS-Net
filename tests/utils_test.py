import os
import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../models')
from data_processor import get_std_mean_from_images

def show_cropped_image(img_batch, mask_batch, img_path, img_format, strategy_type, save_path):
    print('show_cropped_image')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if 'normalize' in strategy_type:
        img_batch = np.moveaxis(img_batch, 1, -1) * 255  # axis move from first channel to last channel
    else:
        std, avg = get_std_mean_from_images(img_path, img_format)
        img_batch = np.moveaxis(img_batch, 1, -1) * std + avg
    mask_batch = np.moveaxis(mask_batch, 1, -1) * 255
    print(img_batch.shape, mask_batch.shape)

    color_index = 0

    index_list = [x for x in range(100)] + [x for x in range(img_batch.shape[0] - 1, img_batch.shape[0] - 401, -1)]

    for batch_index in tqdm(index_list):

        a_mask = img_batch[batch_index, :, :, color_index].astype('uint8')
        save_overlay_img(img_batch[batch_index, :, :, color_index], cv2.Canny(a_mask, 100, 200), save_path, batch_index)

        # cv2.imwrite(save_path + f'image_{batch_index}_{color_index}.png', img_batch[batch_index, :, :, color_index])
        # cv2.imwrite(save_path + f'mask_{batch_index}_{color_index}.png', mask_batch[batch_index, :, :, color_index])


def save_overlay_img(orig_img, mask_edge, save_path, index):
    overlay_edge = orig_img
    overlay_edge[mask_edge>0] = overlay_edge[mask_edge>0] * 0  # clean it before overwriting it
    overlay_edge = overlay_edge + mask_edge
    # overlay_edge[overlay_edge > 255] = 255
    overlay_edge = overlay_edge.astype('uint8')

    cv2.imwrite(save_path + f'overlay_{index}.png', overlay_edge)

    return
