"""
Author: Junbong Jang
Date: 5/31/2021

Store functions that define 2D Keras models
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Activation, Add, Input, concatenate, Conv2D, MaxPooling2D,
AveragePooling2D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose,BatchNormalization, Dropout, GaussianNoise,
GlobalAveragePooling2D, Dense, Flatten, ReLU)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers
from debug_utils import log_function_call
import deep_neural_net_blocks as net_block

@log_function_call
def VGG19D_se(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=[3, img_rows, img_cols])

    a_encoder = net_block.create_VGG19D_encoder(inputs)
    block1_conv, block2_conv, block3_conv, block4_conv, block5_conv = a_encoder(inputs)

    # ------------------ Decoder -----------------------
    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv), block4_conv], axis=1)
    up6 = Dropout(0.5)(up6)
    up6 = net_block.squeeze_and_excitation_block(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv], axis=1)
    up7 = Dropout(0.5)(up7)
    up7 = net_block.squeeze_and_excitation_block(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv], axis=1)
    up8 = Dropout(0.5)(up8)
    up8 = net_block.squeeze_and_excitation_block(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv], axis=1)
    up9 = Dropout(0.5)(up9)
    up9 = net_block.squeeze_and_excitation_block(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    sigmoid_conv = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
        crop_conv = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(sigmoid_conv)
    else:
        # remove reflected portion from the image for prediction
        crop_conv = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(sigmoid_conv)

    model = Model(inputs=inputs, outputs=crop_conv)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model