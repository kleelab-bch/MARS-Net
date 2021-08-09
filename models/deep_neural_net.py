"""
Author: Junbong Jang
Date: 4/30/2020

Store functions that define 2D Keras models
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, ResNet50V2, DenseNet201, InceptionResNetV2
from tensorflow.keras.layers import (Activation, Add, Input, concatenate, Conv2D, MaxPooling2D,
AveragePooling2D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose,BatchNormalization, Dropout, GaussianNoise,
GlobalAveragePooling2D, Dense, Flatten, ReLU)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers
import deep_neural_net_blocks as net_block
from tensorflow.keras.utils import plot_model, get_file
from debug_utils import log_function_call


@log_function_call
def VGG16(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path, encoder_weights='imagenet'):
    inputs = Input(shape=[3, img_rows, img_cols])

    # Create the feature extraction model
    base_model = tf.keras.applications.VGG16(input_shape=[3, img_rows, img_cols], include_top=False, weights=encoder_weights)
    layer_names = [
        'block1_conv2',  # 64x128x128
        'block2_conv2',  # 128x64x64
        'block3_conv3',  # 256x32x32
        'block4_conv3',  # 512x16x16
        'block5_conv3'  # 512x8x8
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    encoder_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    plot_model(encoder_model, to_file='model_plots/encoder_VGG16_train.png', show_shapes=True, show_layer_names=True, dpi=144)

    skips = encoder_model(inputs)

    # upsampling model
    up6 = concatenate([UpSampling2D(size=(2, 2))(skips[4]), skips[3]], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), skips[2]], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), skips[1]], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), skips[0]], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_custom(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=(3,img_rows, img_cols))

    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(conv5)


    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    # to fix the encoder weights and prevent VGG16 training
    #for i in range(18):
    #    model.layers[i].trainable = False

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_dropout(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
    inputs = Input(shape=(3,img_rows, img_cols))

    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(conv5)


    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_l2(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=(3,img_rows, img_cols))

    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv1)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(pool2)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_dac(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=(3,img_rows, img_cols))

    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(conv5)

    dacblock_output = net_block.DACblock(conv5, 512)

    up6 = concatenate([UpSampling2D(size=(2, 2))(dacblock_output), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_spp(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=(3,img_rows, img_cols))

    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(conv5)

    sppblock_output = net_block.spatial_pyramid_pool(conv5)
    #sppblock_output = net_block.SPPblock(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(sppblock_output), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_movie(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=(3, img_rows, img_cols))

    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(conv5)


    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_batchnorm(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=(3,img_rows, img_cols))

    conv1 = net_block.downsample_batch(inputs, 64, (3, 3), name='conv1_1', apply_batchnorm=False)
    conv1 = net_block.downsample_batch(conv1, 64, (3, 3), name='conv1_2')
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = net_block.downsample_batch(pool1, 128, (3, 3), name='conv2_1')
    conv2 = net_block.downsample_batch(conv2, 128, (3, 3), name='conv2_2')
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = net_block.downsample_batch(pool2, 256, (3, 3), name='conv3_1')
    conv3 = net_block.downsample_batch(conv3, 256, (3, 3), name='conv3_2')
    conv3 = net_block.downsample_batch(conv3, 256, (3, 3), name='conv3_3')
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = net_block.downsample_batch(pool3, 512, (3, 3), name='conv4_1')
    conv4 = net_block.downsample_batch(conv4, 512, (3, 3), name='conv4_2')
    conv4 = net_block.downsample_batch(conv4, 512, (3, 3), name='conv4_3')
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = net_block.downsample_batch(pool4, 512, (3, 3), name='conv5_1')
    conv5 = net_block.downsample_batch(conv5, 512, (3, 3), name='conv5_2')
    conv5 = net_block.downsample_batch(conv5, 512, (3, 3), name='conv5_3')


    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = net_block.downsample_batch(up6, 512, (3, 3))
    conv6 = net_block.downsample_batch(conv6, 512, (3, 3))

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = net_block.downsample_batch(up7, 256, (3, 3))
    conv7 = net_block.downsample_batch(conv7, 256, (3, 3))

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = net_block.downsample_batch(up8, 128, (3, 3))
    conv8 = net_block.downsample_batch(conv8, 128, (3, 3))

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = net_block.downsample_batch(up9, 64, (3, 3))
    conv9 = net_block.downsample_batch(conv9, 64, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_instancenorm(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=(3,img_rows, img_cols))

    conv1 = net_block.downsample_instance(inputs, 64, (3, 3), name='conv1_1', apply_instancenorm=False)
    conv1 = net_block.downsample_instance(conv1, 64, (3, 3), name='conv1_2')
    pool1 = MaxPooling2D((2,2), strides=(2,2))(conv1)

    conv2 = net_block.downsample_instance(pool1, 128, (3, 3), name='conv2_1')
    conv2 = net_block.downsample_instance(conv2, 128, (3, 3), name='conv2_2')
    pool2 = MaxPooling2D((2,2), strides=(2,2))(conv2)

    conv3 = net_block.downsample_instance(pool2, 256, (3, 3), name='conv3_1')
    conv3 = net_block.downsample_instance(conv3, 256, (3, 3), name='conv3_2')
    conv3 = net_block.downsample_instance(conv3, 256, (3, 3), name='conv3_3')
    pool3 = MaxPooling2D((2,2), strides=(2,2))(conv3)

    conv4 = net_block.downsample_instance(pool3, 512, (3, 3), name='conv4_1')
    conv4 = net_block.downsample_instance(conv4, 512, (3, 3), name='conv4_2')
    conv4 = net_block.downsample_instance(conv4, 512, (3, 3), name='conv4_3')
    pool4 = MaxPooling2D((2,2), strides=(2,2))(conv4)

    conv5 = net_block.downsample_instance(pool4, 512, (3, 3), name='conv5_1')
    conv5 = net_block.downsample_instance(conv5, 512, (3, 3), name='conv5_2')
    conv5 = net_block.downsample_instance(conv5, 512, (3, 3), name='conv5_3')


    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = net_block.downsample_instance(up6, 512, (3, 3))
    conv6 = net_block.downsample_instance(conv6, 512, (3, 3))

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = net_block.downsample_instance(up7, 256, (3, 3))
    conv7 = net_block.downsample_instance(conv7, 256, (3, 3))

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = net_block.downsample_instance(up8, 128, (3, 3))
    conv8 = net_block.downsample_instance(conv8, 128, (3, 3))

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = net_block.downsample_instance(up9, 64, (3, 3))
    conv9 = net_block.downsample_instance(conv9, 64, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


## --------------------------------------------------------------------------------------

@log_function_call
def VGG19(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path, encoder_weights='imagenet'):
    print('encoder_weights', encoder_weights)
    inputs = Input(shape=[3, img_rows, img_cols])

    # Create the feature extraction model
    base_model = tf.keras.applications.VGG19(input_shape=[3, img_rows, img_cols], include_top=False, weights=encoder_weights)
    layer_names = [
        'block1_conv2',  # 64x128x128
        'block2_conv2',  # 128x64x64
        'block3_conv4',  # 256x32x32
        'block4_conv4',  # 512x16x16
        'block5_conv4'  # 512x8x8
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    encoder_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    plot_model(encoder_model, to_file='model_plots/encoder_VGG19.png', show_shapes=True, show_layer_names=True, dpi=144)

    skips = encoder_model(inputs)

    # decoder
    up6 = concatenate([UpSampling2D(size=(2, 2))(skips[4]), skips[3]], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), skips[2]], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), skips[1]], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), skips[0]], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    elif bottom_crop > 0 and bottom_crop > 0 and right_crop > 0:
        conv10 = Cropping2D(cropping=((crop_margin, bottom_crop), (crop_margin, right_crop)))(conv10)
    else:
        # remove reflected portion from the image for prediction
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(conv10)

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19_imagenet_pretrained(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=[img_rows, img_cols, 3])

    # Create the feature extraction model
    base_model = tf.keras.applications.VGG19(input_shape=[img_rows, img_cols, 3], include_top=False, weights=None)
    layer_names = [
        'block1_conv2',  # 64x128x128
        'block2_conv2',  # 128x64x64
        'block3_conv4',  # 256x32x32
        'block4_conv4',  # 512x16x16
        'block5_conv4'  # 512x8x8
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    encoder_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    plot_model(encoder_model, to_file='model_plots/encoder_VGG19.png', show_shapes=True, show_layer_names=True, dpi=144)

    skips = encoder_model(inputs)

    # upsampling model
    up6 = concatenate([UpSampling2D(size=(2, 2))(skips[4]), skips[3]], axis=-1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), skips[2]], axis=-1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), skips[1]], axis=-1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), skips[0]], axis=-1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19_freeze(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path, encoder_weights='imagenet'):
    print('encoder_weights', encoder_weights)
    inputs = Input(shape=[3, img_rows, img_cols])

    # Create the feature extraction model
    base_model = tf.keras.applications.VGG19(input_shape=[3, img_rows, img_cols], include_top=False, weights=encoder_weights)
    layer_names = [
        'block1_conv2',  # 64x128x128
        'block2_conv2',  # 128x64x64
        'block3_conv4',  # 256x32x32
        'block4_conv4',  # 512x16x16
        'block5_conv4'  # 512x8x8
    ]

    for i in range(21):  # first 21 layers of the pre-trained VGG model
        base_model.layers[i].trainable = False

    layers = [base_model.get_layer(name).output for name in layer_names]

    encoder_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    plot_model(encoder_model, to_file='model_plots/encoder_VGG19.png', show_shapes=True, show_layer_names=True, dpi=144)

    skips = encoder_model(inputs)

    # upsampling model
    up6 = concatenate([UpSampling2D(size=(2, 2))(skips[4]), skips[3]], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), skips[2]], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), skips[1]], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), skips[0]], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model



@log_function_call
def VGG19_batchnorm_dropout(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # mimic model in https://github.com/clguo/SA-UNet/blob/844808189526afaf06296ba6c135b9c0ba67d70a/SA_UNet.py
    inputs = Input(shape=(3, img_rows, img_cols))

    conv1 = net_block.downsample_batch_dropout(inputs, 64, (3, 3), name='block1_conv1')
    conv1 = net_block.downsample_batch_dropout(conv1, 64, (3, 3), name='block1_conv2')
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    conv2 = net_block.downsample_batch_dropout(pool1, 128, (3, 3), name='block2_conv1')
    conv2 = net_block.downsample_batch_dropout(conv2, 128, (3, 3), name='block2_conv2')
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    conv3 = net_block.downsample_batch_dropout(pool2, 256, (3, 3), name='block3_conv1')
    conv3 = net_block.downsample_batch_dropout(conv3, 256, (3, 3), name='block3_conv2')
    conv3 = net_block.downsample_batch_dropout(conv3, 256, (3, 3), name='block3_conv3')
    conv3 = net_block.downsample_batch_dropout(conv3, 256, (3, 3), name='block3_conv4')
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    conv4 = net_block.downsample_batch_dropout(pool3, 512, (3, 3), name='block4_conv1')
    conv4 = net_block.downsample_batch_dropout(conv4, 512, (3, 3), name='block4_conv2')
    conv4 = net_block.downsample_batch_dropout(conv4, 512, (3, 3), name='block4_conv3')
    conv4 = net_block.downsample_batch_dropout(conv4, 512, (3, 3), name='block4_conv4')
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    conv5 = net_block.downsample_batch_dropout(pool4, 512, (3, 3), name='block5_conv1')
    conv5 = net_block.downsample_batch_dropout(conv5, 512, (3, 3), name='block5_conv2')
    conv5 = net_block.downsample_batch_dropout(conv5, 512, (3, 3), name='block5_conv3')
    conv5 = net_block.downsample_batch_dropout(conv5, 512, (3, 3), name='block5_conv4')


    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = net_block.downsample_batch_dropout(up6, 512, (3, 3))
    conv6 = net_block.downsample_batch_dropout(conv6, 512, (3, 3))

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = net_block.downsample_batch_dropout(up7, 256, (3, 3))
    conv7 = net_block.downsample_batch_dropout(conv7, 256, (3, 3))

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = net_block.downsample_batch_dropout(up8, 128, (3, 3))
    conv8 = net_block.downsample_batch_dropout(conv8, 128, (3, 3))

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = net_block.downsample_batch_dropout(up9, 64, (3, 3))
    conv9 = net_block.downsample_batch_dropout(conv9, 64, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    # Load weights.
    if weights_path == '':
        WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/vgg19/'
                               'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19_batchnorm(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path, encoder_weights='imagenet'):
    inputs = Input(shape=(3, img_rows, img_cols))

    conv1 = net_block.downsample_batch(inputs, 64, (3, 3), name='block1_conv1', apply_batchnorm=False)
    conv1 = net_block.downsample_batch(conv1, 64, (3, 3), name='block1_conv2')
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    conv2 = net_block.downsample_batch(pool1, 128, (3, 3), name='block2_conv1')
    conv2 = net_block.downsample_batch(conv2, 128, (3, 3), name='block2_conv2')
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    conv3 = net_block.downsample_batch(pool2, 256, (3, 3), name='block3_conv1')
    conv3 = net_block.downsample_batch(conv3, 256, (3, 3), name='block3_conv2')
    conv3 = net_block.downsample_batch(conv3, 256, (3, 3), name='block3_conv3')
    conv3 = net_block.downsample_batch(conv3, 256, (3, 3), name='block3_conv4')
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    conv4 = net_block.downsample_batch(pool3, 512, (3, 3), name='block4_conv1')
    conv4 = net_block.downsample_batch(conv4, 512, (3, 3), name='block4_conv2')
    conv4 = net_block.downsample_batch(conv4, 512, (3, 3), name='block4_conv3')
    conv4 = net_block.downsample_batch(conv4, 512, (3, 3), name='block4_conv4')
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    conv5 = net_block.downsample_batch(pool4, 512, (3, 3), name='block5_conv1')
    conv5 = net_block.downsample_batch(conv5, 512, (3, 3), name='block5_conv2')
    conv5 = net_block.downsample_batch(conv5, 512, (3, 3), name='block5_conv3')
    conv5 = net_block.downsample_batch(conv5, 512, (3, 3), name='block5_conv4')


    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = net_block.downsample_batch(up6, 512, (3, 3))
    conv6 = net_block.downsample_batch(conv6, 512, (3, 3))

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = net_block.downsample_batch(up7, 256, (3, 3))
    conv7 = net_block.downsample_batch(conv7, 256, (3, 3))

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = net_block.downsample_batch(up8, 128, (3, 3))
    conv8 = net_block.downsample_batch(conv8, 128, (3, 3))

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = net_block.downsample_batch(up9, 64, (3, 3))
    conv9 = net_block.downsample_batch(conv9, 64, (3, 3))

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    # Load weights.
    if weights_path == '':
        WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/vgg19/'
                               'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19D_crop_first(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)
    x = Dropout(0.5)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)
    x = Dropout(0.5)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)


    # upsampling model
    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv4), block4_conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv4], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv2], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    if bottom_crop == 0:
        crop_conv = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(conv9)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        crop_conv = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(conv9)  # remove reflected portion from the image for prediction

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(crop_conv)

    model = Model(inputs=inputs, outputs=conv10)

    # Load weights.
    if weights_path == '':
        WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/vgg19/'
                               'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19_dropout(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/vgg19.py#L45-L230
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)
    x = Dropout(0.5)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)
    x = Dropout(0.5)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)


    # upsampling model
    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv4), block4_conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv4], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv2], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    # Load weights.
    if weights_path == '':
        WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/vgg19/'
                               'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19_dropout_gelu(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/vgg19.py#L45-L230
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)
    x = Dropout(0.5)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)
    x = Dropout(0.5)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same', name='block5_conv4')(x)


    # upsampling model
    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv4), block4_conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation=tf.keras.activations.gelu, padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv4], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation=tf.keras.activations.gelu, padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation=tf.keras.activations.gelu, padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation=tf.keras.activations.gelu, padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation=tf.keras.activations.gelu, padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv2], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation=tf.keras.activations.gelu, padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation=tf.keras.activations.gelu, padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    # Load weights.
    if weights_path == '':
        WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/vgg19/'
                               'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19_dropout_swish(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/vgg19.py#L45-L230
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)
    x = Dropout(0.5)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)
    x = Dropout(0.5)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same', name='block5_conv4')(x)


    # upsampling model
    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv4), block4_conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation=tf.keras.activations.swish, padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv4], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation=tf.keras.activations.swish, padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation=tf.keras.activations.swish, padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation=tf.keras.activations.swish, padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation=tf.keras.activations.swish, padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv2], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation=tf.keras.activations.swish, padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation=tf.keras.activations.swish, padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    # Load weights.
    if weights_path == '':
        WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/vgg19/'
                               'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19_dropout_feature_extractor(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/applications/vgg19.py#L45-L230
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)
    x = Dropout(0.5)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)
    x = Dropout(0.5)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

    global_pooled_conv5 = GlobalAveragePooling2D(data_format='channels_first', name='style_output')(block5_conv4)

    # upsampling model
    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv4), block4_conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv4], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv2], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=[conv10, global_pooled_conv5])

    # Load weights.
    if weights_path == '':
        WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/vgg19/'
                               'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)


    return model


@log_function_call
def VGG19_dropout_dac(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)
    x = Dropout(0.5)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)
    x = Dropout(0.5)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

    dacblock_output = net_block.DACblock(block5_conv4, 512)

    # decoder
    up6 = concatenate([UpSampling2D(size=(2, 2))(dacblock_output), block4_conv4], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv4], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv2], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv2], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    # Load weights.
    if weights_path == '':
        WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                               'keras-applications/vgg19/'
                               'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

        weights_path = get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
    model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def UNet(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model



@log_function_call
def UNet_small(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def UNet_imagenet_pretrained(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def UNet_feature_extractor(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    global_pooled_conv5 = GlobalAveragePooling2D(data_format='channels_first',name='style_output')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin),(crop_margin, crop_margin)), name='segmentation_output')(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop),(0, right_crop)), name='segmentation_output')(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=[conv10, global_pooled_conv5])

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model

## --------------------------------------------------------------------------------------
@log_function_call
def ResNet50V2Keras(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input((3, img_rows, img_cols))
    conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(inputs)

    baseline_model = ResNet50V2(input_tensor=inputs, include_top=False, weights='imagenet')
    conv5 = baseline_model.output

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), baseline_model.get_layer(name='conv4_block5_out').output],
                      axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), baseline_model.get_layer(name='conv3_block3_out').output],
                      axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), baseline_model.get_layer(name='conv2_block2_out').output],
                      axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), baseline_model.get_layer(name='conv1_conv').output], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv1], axis=1)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv10)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)

    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10, name='ResNet50V2')

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


# @log_function_call
# def ResNet50V2Keras(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
#     baseline_model = ResNet50V2(input_shape=(3, img_rows, img_cols), include_top=False, weights='imagenet')
#     conv5 = baseline_model.output
#
#     # activation_40,22,10,1
#     up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), baseline_model.get_layer(name='conv4_block5_out').output],
#                       axis=1)
#     conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
#
#     up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), baseline_model.get_layer(name='conv3_block3_out').output],
#                       axis=1)
#     conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
#
#     up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), baseline_model.get_layer(name='conv2_block2_out').output],
#                       axis=1)
#     conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
#
#     up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), baseline_model.get_layer(name='conv1_conv').output], axis=1)
#     conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
#
#     up10 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv9))
#     conv10 = Conv2D(1, (1, 1), activation='sigmoid')(up10)
#
#     if bottom_crop == 0:
#         conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
#             conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
#     else:
#         conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
#             conv10)  # remove reflected portion from the image for prediction
#
#     model = Model(inputs=baseline_model.input, outputs=conv10, name='ResNet50V2')
#
#     if weights_path != '':
#         model.load_weights(weights_path, by_name=True)
#
#     return model


@log_function_call
def InceptionResV2(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    baseline_model = InceptionResNetV2(input_shape=(3, img_rows, img_cols), include_top=False, weights='imagenet')

    print(baseline_model.summary())
    plot_model(baseline_model, to_file='model_plots/encoder_InceptionResV2_train.png', show_shapes=True,
               show_layer_names=True, dpi=144)

    conv5 = baseline_model.output

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), baseline_model.get_layer(name='conv4_block5_out').output],
                      axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), baseline_model.get_layer(name='conv3_block3_out').output],
                      axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), baseline_model.get_layer(name='conv2_block2_out').output],
                      axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), baseline_model.get_layer(name='conv1_conv').output], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    up10 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv9))
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(up10)

    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=baseline_model.input, outputs=conv10, name='ResNet50V2')

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def ResBiT(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # not used due to inability to load pertrained model
    import tensorflow_hub as hub
    baseline_model = hub.KerasLayer("https://tfhub.dev/google/bit/m-r152x4/1", input_shape=(3, img_rows, img_cols))
    conv5 = baseline_model.output
    # activation_40,22,10,1
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), baseline_model.get_layer(name='conv4_block5_out').output],
                      axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), baseline_model.get_layer(name='conv3_block3_out').output],
                      axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), baseline_model.get_layer(name='conv2_block2_out').output],
                      axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), baseline_model.get_layer(name='conv1_conv').output], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    up10 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv9))
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(up10)

    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=baseline_model.input, outputs=conv10, name='ResBiT')

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


# https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff
@log_function_call
def ResNet50JJ(img_rows, img_cols, crop_margin, right_crop, bottom_crop, is_training, weights_path):
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=(3, img_rows, img_cols))

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=1, name='bn_conv1', trainable=is_training)(X)
    conv1 = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    # Stage 2
    X = net_block.convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1, is_training=is_training)
    X = net_block.identity_block(X, 3, [64, 64, 256], stage=2, block='b', is_training=is_training)
    conv2 = net_block.identity_block(X, 3, [64, 64, 256], stage=2, block='c', is_training=is_training)

    # Stage 3
    X = net_block.convolutional_block(conv2, f=3, filters=[128, 128, 512], stage=3, block='a', s=2, is_training=is_training)
    X = net_block.identity_block(X, 3, [128, 128, 512], stage=3, block='b', is_training=is_training)
    X = net_block.identity_block(X, 3, [128, 128, 512], stage=3, block='c', is_training=is_training)
    conv3 = net_block.identity_block(X, 3, [128, 128, 512], stage=3, block='d', is_training=is_training)

    # Stage 4
    X = net_block.convolutional_block(conv3, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2, is_training=is_training)
    X = net_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='b', is_training=is_training)
    X = net_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='c', is_training=is_training)
    X = net_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='d', is_training=is_training)
    X = net_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='e', is_training=is_training)
    conv4 = net_block.identity_block(X, 3, [256, 256, 1024], stage=4, block='f', is_training=is_training)

    # Stage 5
    X = net_block.convolutional_block(conv4, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2, is_training=is_training)
    X = net_block.identity_block(X, 3, [512, 512, 2048], stage=5, block='b', is_training=is_training)
    conv5 = net_block.identity_block(X, 3, [512, 512, 2048], stage=5, block='c', is_training=is_training)

    # ------------ Expansion -----------------
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    up10 = UpSampling2D(size=(2, 2))(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(up10)

    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=X_input, outputs=conv10, name='ResNet50')

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def DenseNet201Keras(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    baseline_model = DenseNet201(input_shape=(3, img_rows, img_cols), include_top=False, weights='imagenet')
    conv5 = baseline_model.output

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), baseline_model.get_layer(name='pool4_conv').output], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), baseline_model.get_layer(name='pool3_conv').output], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), baseline_model.get_layer(name='pool2_conv').output], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), baseline_model.get_layer(name='conv1/relu').output], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    up10 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv9))
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(up10)

    if bottom_crop == 0:
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(
            conv10)  # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(
            conv10)  # remove reflected portion from the image for prediction

    # Create model
    model = Model(inputs=baseline_model.input, outputs=conv10, name='DenseNet201')

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def EFF_B7(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # note that this is last channel
    inputs = Input(shape=[img_rows, img_cols, 3])

    # Create the feature extraction model
    base_model = tf.keras.applications.EfficientNetB7(input_shape=[img_rows, img_cols, 3], include_top=False, weights='imagenet')
    layer_names = [
        'block1d_add',  # 64x64x32
        'block2g_add',  # 32x32x48
        'block3g_add',  # 16x16x80
        'block5j_add',  # 8x8x224
        'block7d_add'  # 4x4x640
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    encoder_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    plot_model(encoder_model, to_file='model_plots/encoder_EFF_B7_train.png', show_shapes=True,show_layer_names=True, dpi=300)
    skips = encoder_model(inputs)

    # -------------decoder--------------
    up6 = concatenate([UpSampling2D(size=(2, 2))(skips[4]), skips[3]], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), skips[2]], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), skips[1]], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), skips[0]], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    up10 = UpSampling2D(size=(2, 2))(conv9)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv10)

    conv11 = Conv2D(1, (1, 1), activation='sigmoid')(conv10)
    if bottom_crop == 0:
        # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
        conv11 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(conv11)
    else:
        # remove reflected portion from the image for prediction
        conv11 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(conv11)

    model = Model(inputs=inputs, outputs=conv11)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model
