"""
Author: Junbong Jang
Date: 6/9/2020

Store functions that define deep learning MTL models
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, ResNet50V2, DenseNet201, InceptionResNetV2
from tensorflow.keras.layers import (Layer, Activation, Add, Input, concatenate, Conv2D, MaxPooling2D,
AveragePooling2D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose,BatchNormalization, Dropout, GaussianNoise,
GlobalAveragePooling2D, Dense, Flatten, ReLU)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2
import deep_neural_net_blocks as net_block
from tensorflow.keras.utils import plot_model, get_file
from debug_utils import log_function_call

from deep_neural_net_layer import *


@log_function_call
def VGG19_classifier_custom_loss(img_rows, img_cols, weights_path):
    # This function is to test if custom loss layer works... I empirically confirmed that this works
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

    x = GlobalAveragePooling2D()(block5_conv4)
    output = Dense(1, activation='sigmoid', name='classifier')(x)

    # model = Model(inputs=inputs, outputs=output)

    y_true = Input(shape=[1, ], name='y_true')
    out = CustomLossLayer()(y_true, output)
    model = Model([inputs, y_true], out)

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
def VGG19_classifier_regressor(img_rows, img_cols, weights_path):
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

    x = GlobalAveragePooling2D()(block5_conv4)
    classification_output = Dense(1, activation='sigmoid', name='classifier')(x)
    regression_output = Dense(1, name='regressor')(x)

    model = Model(inputs=inputs, outputs=[regression_output, classification_output])

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
def VGG19_MTL(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

    x = GlobalAveragePooling2D()(block5_conv4)
    classification_output = Dense(1, activation='sigmoid', name='classifier')(x)
    regression_output = Dense(1, name='regressor')(x)

    # ----------------- segmentation decoder -----------------
    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv4), block4_conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv4], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv2], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    if bottom_crop == 0:
        # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)), name='segmentation')(conv10)
    else:
        # remove reflected portion from the image for prediction
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)), name='segmentation')(conv10)

    # ----------------- autoencoder decoder -----------------
    aut_conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    aut_conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(aut_conv6)

    aut_up7 = concatenate([UpSampling2D(size=(2, 2))(aut_conv6), block3_conv4], axis=1)
    aut_conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(aut_up7)
    aut_conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(aut_conv7)

    aut_up8 = concatenate([UpSampling2D(size=(2, 2))(aut_conv7), block2_conv2], axis=1)
    aut_conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(aut_up8)
    aut_conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(aut_conv8)

    aut_up9 = concatenate([UpSampling2D(size=(2, 2))(aut_conv8), block1_conv2], axis=1)
    aut_conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(aut_up9)
    aut_conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(aut_conv9)

    aut_conv10 = Conv2D(1, (1, 1), activation='linear', name='autoencoder')(aut_conv9)

    model = Model(inputs=inputs, outputs=[conv10, aut_conv10, regression_output, classification_output])

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
def VGG19_MTL_auto(img_rows, img_cols, crop_margin, right_crop, bottom_crop, removed_tasks, weights_path):
    inputs = Input(shape=[3, img_rows, img_cols])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

    x = GlobalAveragePooling2D()(block5_conv4)
    classification_output = Dense(1, activation='sigmoid', name='classifier')(x)
    regression_output = Dense(1, name='regressor')(x)

    # decoder
    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv4), block4_conv4], axis=1)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv4], axis=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv2], axis=1)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv2], axis=1)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
        conv10 = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)), name='segmentation')(conv10)
    else:
        # remove reflected portion from the image for prediction
        conv10 = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)), name='segmentation')(conv10)

    # ----------------- autoencoder decoder -----------------
    aut_conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    aut_conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(aut_conv6)

    aut_up7 = concatenate([UpSampling2D(size=(2, 2))(aut_conv6), block3_conv4], axis=1)
    aut_conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(aut_up7)
    aut_conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(aut_conv7)

    aut_up8 = concatenate([UpSampling2D(size=(2, 2))(aut_conv7), block2_conv2], axis=1)
    aut_conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(aut_up8)
    aut_conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(aut_conv8)

    aut_up9 = concatenate([UpSampling2D(size=(2, 2))(aut_conv8), block1_conv2], axis=1)
    aut_conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(aut_up9)
    aut_conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(aut_conv9)

    aut_conv10 = Conv2D(1, (1, 1), activation='linear', name='autoencoder')(aut_conv9)

    y1_true = Input(shape=[1, img_rows - 60, img_cols - 60], name='y1_true')
    y2_true = Input(shape=[3, img_rows, img_cols], name='y2_true')
    y3_true = Input(shape=[1, ], name='y3_true')
    y4_true = Input(shape=[1, ], name='y4_true')

    out = CustomMultiLossLayer(removed_tasks, nb_outputs=4)([y1_true, y2_true, y3_true, y4_true, conv10, aut_conv10, regression_output, classification_output])
    model = Model([inputs, y1_true, y2_true, y3_true, y4_true], out)

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