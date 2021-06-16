'''
Junbong Jang
6/7/2021

CAM for image classifier
'''
import sys
sys.path.append('../models')
from data_processor import preprocess_input

import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import cv2
from tqdm import tqdm


def visualize_feature_activation_map(model, input_images, image_names, save_path):
    '''
    input_images should be unprocessed

    Returns the passed in image with a heatmap of the specific feature number overlaid
    reference https://codeocean.com/capsule/0685076/tree/v1

    '''

    def channel_average_pool_conv(conv_feature_map):
        # input: conv_feature_map 8 x 8 x 512
        # output: averaged_feature_map 8 x 8

        averaged_feature_map = np.zeros(conv_feature_map.shape[:2], dtype=np.float32)
        for i in range(conv_feature_map.shape[0]):
            for j in range(conv_feature_map.shape[1]):
                for k in range(conv_feature_map.shape[2]):
                    averaged_feature_map[i, j] = averaged_feature_map[i, j] + conv_feature_map[i, j, k]
        return averaged_feature_map

    print('visualize_feature_activation_map')
    print(input_images.shape, len(image_names), image_names[0])

    for i, image_name in enumerate(tqdm(image_names)):
        image = preprocess_input(input_images[i])
        channels, width, height = image.shape

        # Get feature map from the model
        final_conv_layer = model.get_layer(name='block5_conv4')
        get_output = K.function([model.layers[0].input], [final_conv_layer.output])
        conv_output = np.squeeze(get_output(np.expand_dims(image, axis=0))[0])  # 512 x 8 x 8
        dense_weights = model.get_layer(name='dense').get_weights()[0]  # 512 x 1
        dense_weights = np.expand_dims(dense_weights, axis=-1)  # 512 x 1 x 1

        # generate heatmap
        cam = tf.math.multiply(conv_output, dense_weights)  # 512 x 8 x 8
        cam = channel_average_pool_conv(np.moveaxis(cam, 0, -1))  # 8 x 8
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, (height, width))  # 128x128
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_HOT)  # 128x128x3
        # print(image.shape, image.dtype, conv_output.shape, cam.shape, heatmap.shape, heatmap.dtype)

        # overlay heatmap on original image
        alpha = 0.85
        heatmap_img = cv2.addWeighted(np.uint8(np.moveaxis(input_images[i], 0, -1)), alpha, heatmap, 1 - alpha, 0)  # 128x128x3

        # image_name = image_name.replace('../crop/generated/crop_even_FNA_test/img_repeat0/', '')
        cv2.imwrite(f'{save_path}heatmap_{image_name}', heatmap_img)


