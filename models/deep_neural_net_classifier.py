"""
Author: Junbong Jang
Date: 6/9/2020

Store functions that define deep learning classifiers
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, ResNet50V2, DenseNet201, InceptionResNetV2
from tensorflow.keras.layers import (Activation, Add, Input, concatenate, Conv2D, MaxPooling2D,
AveragePooling2D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose,BatchNormalization, Dropout, GaussianNoise,
GlobalAveragePooling2D, Dense, Flatten, ReLU)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2
import deep_neural_net_blocks as net_block
from tensorflow.keras.utils import plot_model, get_file
from debug_utils import log_function_call


@log_function_call
def VGG19_classifier(img_rows, img_cols, weights_path):
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
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)

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
def UNet_encoder_classifier(img_rows, img_cols, weights_path):
    inputs = Input(shape=[img_rows, img_cols, 3])

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', kernel_regularizer=l2(0.0005))(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    output = Dense(1000, activation='softmax', kernel_regularizer=l2(0.0005))(x)

    model = Model(inputs=inputs, outputs=output)

    # Load weights.
    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG16_classifier(img_rows, img_cols, weights_path):
    inputs = Input(shape=[img_rows, img_cols, 3])

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', kernel_regularizer=l2(0.0005))(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', kernel_regularizer=l2(0.0005))(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = Dropout(0.5)(x)
    output = Dense(1000, activation='softmax', kernel_regularizer=l2(0.0005))(x)

    model = Model(inputs=inputs, outputs=output)

    # from tensorflow.python.keras.utils import data_utils
    # WEIGHTS_PATH = ('https://storage.googleapis.com/tensorflow/keras-applications/'
    #                 'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    # weights_path = data_utils.get_file(
    #     'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
    #     WEIGHTS_PATH,
    #     cache_subdir='models',
    #     file_hash='64373286793e3c8b2b4e3219cbf3544b')
    #
    # model.load_weights(weights_path)

    # Load weights.
    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19D_classifier(img_rows, img_cols, weights_path):
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

    x = GlobalAveragePooling2D()(block5_conv4)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)

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
def EFF_B7_classifier(img_rows, img_cols, weights_path):
    inputs = Input(shape=[img_rows, img_cols, 3])

    model = tf.keras.applications.EfficientNetB7(input_shape=[img_rows, img_cols, 3], include_top=False,
                                                 weights='imagenet')
    x = model(inputs)
    x = GlobalAveragePooling2D()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model
