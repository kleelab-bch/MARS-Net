"""
Author: Junbong Jang
Date: 4/30/2021

Store functions that define 3D Keras models
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, concatenate,
                                     Conv2D, MaxPooling2D, UpSampling2D, Cropping2D,
                                     Conv3D, MaxPooling3D, UpSampling3D, Cropping3D, Dropout)
from tensorflow.keras.utils import get_file

from debug_utils import log_function_call


@log_function_call
def UNet_3D(img_depth, img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    inputs = Input((3, img_depth, img_rows, img_cols))
    conv1 = Conv3D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = Conv3D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(filters=128, kernel_size=3, activation='relu', padding='same')(pool1)
    conv2 = Conv3D(filters=128, kernel_size=3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(filters=256, kernel_size=3, activation='relu', padding='same')(pool2)
    conv3 = Conv3D(filters=256, kernel_size=3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(filters=512, kernel_size=3, activation='relu', padding='same')(pool3)
    conv4 = Conv3D(filters=512, kernel_size=3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(filters=1024, kernel_size=3, activation='relu', padding='same')(pool4)
    conv5 = Conv3D(filters=1024, kernel_size=3, activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv4], axis=1)
    conv6 = Conv3D(filters=512, kernel_size=3, activation='relu', padding='same')(up6)
    conv6 = Conv3D(filters=512, kernel_size=3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv3], axis=1)
    conv7 = Conv3D(filters=256, kernel_size=3, activation='relu', padding='same')(up7)
    conv7 = Conv3D(filters=256, kernel_size=3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling3D(size=(2, 2, 2))(conv7), conv2], axis=1)
    conv8 = Conv3D(filters=128, kernel_size=3, activation='relu', padding='same')(up8)
    conv8 = Conv3D(filters=128, kernel_size=3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling3D(size=(2, 2, 2))(conv8), conv1], axis=1)
    conv9 = Conv3D(filters=64, kernel_size=3, activation='relu', padding='same')(up9)
    conv9 = Conv3D(filters=64, kernel_size=3, activation='relu', padding='same')(conv9)

    conv10 = Conv3D(filters=1, kernel_size=(1, 1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        conv10 = Cropping3D(cropping=((0, 0),(crop_margin, crop_margin),(crop_margin, crop_margin)))(conv10) # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
    else:
        conv10 = Cropping3D(cropping=((0, 0),(0, bottom_crop),(0, right_crop)))(conv10)  # remove reflected portion from the image for prediction

    model = Model(inputs=inputs, outputs=conv10)

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


def VGG19D_attn_temporal(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):

    def create_encoder(inputs):
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

        return tf.keras.Model(inputs=[inputs], outputs=[block1_conv2, block2_conv2, block3_conv4, block4_conv4, block5_conv4])

    def temporal_attn_block(cur_frame_conv, M1_conv, M2_conv, M3_conv, M4_conv, value_channel_num, width_height):
        def encoding_layers(a_conv, channel_num):
            a_conv = Conv2D(channel_num, (1, 1), activation='relu', padding='same')(a_conv)
            a_conv = Conv2D(channel_num, (3, 3), activation='relu', padding='same')(a_conv)
            return a_conv

        print('temporal_attn_block')
        key_channel_num = value_channel_num//4
        memory_num = 4

        cur_frame_key = encoding_layers(cur_frame_conv, key_channel_num)
        cur_frame_value = encoding_layers(cur_frame_conv, value_channel_num)

        M1_key = encoding_layers(M1_conv, key_channel_num)
        M1_value = encoding_layers(M1_conv, value_channel_num)
        M2_key = encoding_layers(M2_conv, key_channel_num)
        M2_value = encoding_layers(M2_conv, value_channel_num)
        M3_key = encoding_layers(M3_conv, key_channel_num)
        M3_value = encoding_layers(M3_conv, value_channel_num)
        M4_key = encoding_layers(M4_conv, key_channel_num)
        M4_value = encoding_layers(M4_conv, value_channel_num)

        M_key = concatenate([tf.expand_dims(M1_key, axis=1),
                    tf.expand_dims(M2_key, axis=1),
                    tf.expand_dims(M3_key, axis=1),
                    tf.expand_dims(M4_key, axis=1)], axis=1)
        M_value = concatenate([tf.expand_dims(M1_value, axis=1),
                    tf.expand_dims(M2_value, axis=1),
                    tf.expand_dims(M3_value, axis=1),
                    tf.expand_dims(M4_value, axis=1)], axis=1)
        print(cur_frame_key.shape, cur_frame_value.shape, M_key.shape, M_value.shape)
        print('---------')

        # ------------------- Calculate temporal memory attention ---------------------
        # cur_frame_value: C x H x W
        # cur_frame_key: C x H x W --> C x HW --> HW x C
        cur_frame_key = tf.reshape(cur_frame_key, shape=(-1, key_channel_num, width_height*width_height))
        print(cur_frame_key.shape) # (None, 128, 64)
        cur_frame_key = tf.transpose(cur_frame_key, perm=[0,2,1])
        print(cur_frame_key.shape) # (None, 64, 128)
        # M_key: T x C x H x W --> C x T x H x W --> C x THW
        M_key = tf.transpose(M_key, perm=[0,2,1,3,4])
        print(M_key.shape) # (None, 128, 4, 8, 8)
        M_key = tf.reshape(M_key, shape=(-1, key_channel_num, memory_num*width_height*width_height))
        print(M_key.shape) # (None, 128, 256)
        # M_value: T x C x H x W --> T x H x W x C --> THW x C
        M_value = tf.transpose(M_value, perm=[0,1,3,4,2])
        print(M_value.shape) # (None, 4, 8, 8, 512)
        M_value = tf.reshape(M_value, shape=(-1, memory_num*width_height*width_height, value_channel_num))
        print(M_value.shape) # (None, 256, 512)

        # HW x C and C x THW --> HW x THW
        softmax = tf.nn.softmax(tf.matmul(cur_frame_key, M_key))
        print(softmax.shape) # (None, 64, 256)
        # HW x THW and THW x C --> HW x C
        memory_attn = tf.matmul(softmax, M_value)
        # HW x C --> C x HW --> C x H x W
        print(memory_attn.shape) # (None, 64, 512)
        memory_attn = tf.transpose(memory_attn, perm=[0,2,1])
        print(memory_attn.shape) # (None, 512, 64)
        memory_attn_reshaped = tf.reshape(memory_attn, shape=(-1, value_channel_num, width_height, width_height))
        print(memory_attn_reshaped.shape) # (None, 512, 8, 8)

        cur_frame_value_and_memory_attn = concatenate([cur_frame_value, memory_attn_reshaped], axis=1)
        print(cur_frame_value_and_memory_attn.shape)
        return cur_frame_value_and_memory_attn

    # ------------------------ Model Creation -----------------------------------
    # M stands for Memory, M1 means previous 1 frame from current frame
    cur_frame_input = Input(shape=[3, img_rows, img_cols])
    M1_input = Input(shape=[3, img_rows, img_cols])
    M2_input = Input(shape=[3, img_rows, img_cols])
    M3_input = Input(shape=[3, img_rows, img_cols])
    M4_input = Input(shape=[3, img_rows, img_cols])

    a_encoder = create_encoder(cur_frame_input)
    block1_conv, block2_conv, block3_conv, block4_conv, block5_conv = a_encoder(cur_frame_input)
    M1_block1_conv, M1_block2_conv, M1_block3_conv, M1_block4_conv, M1_block5_conv = a_encoder(M1_input)
    M2_block1_conv, M2_block2_conv, M2_block3_conv, M2_block4_conv, M2_block5_conv = a_encoder(M2_input)
    M3_block1_conv, M3_block2_conv, M3_block3_conv, M3_block4_conv, M3_block5_conv = a_encoder(M3_input)
    M4_block1_conv, M4_block2_conv, M4_block3_conv, M4_block4_conv, M4_block5_conv = a_encoder(M4_input)

    block5_conv = temporal_attn_block(block5_conv, M1_block5_conv, M2_block5_conv, M3_block5_conv, M4_block5_conv, 512, 8)

    # ------------------ Decoder -----------------------

    up6 = concatenate([UpSampling2D(size=(2, 2))(block5_conv), block4_conv], axis=1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), block3_conv], axis=1)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), block2_conv], axis=1)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), block1_conv], axis=1)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    sigmoid_conv = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    if bottom_crop == 0:
        # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
        crop_conv = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(sigmoid_conv)
    else:
        # remove reflected portion from the image for prediction
        crop_conv = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(sigmoid_conv)

    model = Model(inputs=[cur_frame_input, M1_input, M2_input, M3_input, M4_input], outputs=[crop_conv])

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
