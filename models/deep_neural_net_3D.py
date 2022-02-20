"""
Author: Junbong Jang
Date: 4/30/2021

Store functions that define 3D Keras models
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, concatenate, Subtract, Add, Multiply,
                                     Conv2D, MaxPooling2D, UpSampling2D, Cropping2D,
                                     Conv3D, MaxPooling3D, UpSampling3D, Cropping3D, Dropout)
from tensorflow.keras.utils import get_file

import deep_neural_net_blocks as net_block
from debug_utils import log_function_call
import tensorflow.keras.backend as K
import math

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


@log_function_call
def VGG19D_temporal_attn(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # ------------------------ Model Creation -----------------------------------
    # M stands for Memory, M1 means previous 1 frame from current frame
    cur_frame_input = Input(shape=[3, img_rows, img_cols])
    M1_input = Input(shape=[3, img_rows, img_cols])
    M2_input = Input(shape=[3, img_rows, img_cols])
    M3_input = Input(shape=[3, img_rows, img_cols])
    M4_input = Input(shape=[3, img_rows, img_cols])

    a_encoder = net_block.create_VGG19D_encoder(cur_frame_input)
    block1_conv, block2_conv, block3_conv, block4_conv, block5_conv = a_encoder(cur_frame_input)
    M1_block1_conv, M1_block2_conv, M1_block3_conv, M1_block4_conv, M1_block5_conv = a_encoder(M1_input)
    M2_block1_conv, M2_block2_conv, M2_block3_conv, M2_block4_conv, M2_block5_conv = a_encoder(M2_input)
    M3_block1_conv, M3_block2_conv, M3_block3_conv, M3_block4_conv, M3_block5_conv = a_encoder(M3_input)
    M4_block1_conv, M4_block2_conv, M4_block3_conv, M4_block4_conv, M4_block5_conv = a_encoder(M4_input)

    block4_conv = net_block.temporal_attn_block(block4_conv, M1_block4_conv, M2_block4_conv, M3_block4_conv, M4_block4_conv, 256)
    block5_conv = net_block.temporal_attn_block(block5_conv, M1_block5_conv, M2_block5_conv, M3_block5_conv, M4_block5_conv, 256)

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

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19D_temporal_attn_v2(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):

    def temporal_attn_block(cur_frame_conv, M1_conv, M2_conv, M3_conv, M4_conv):
        print('temporal_attn_block')

        M_concat = concatenate([tf.expand_dims(M1_conv, axis=1),
                                 tf.expand_dims(M2_conv, axis=1),
                                 tf.expand_dims(M3_conv, axis=1),
                                 tf.expand_dims(M4_conv, axis=1)], axis=1)

        # convolve 1x1 to reduce their channels to equal channels of (cur_frame_conv)
        temporal_attention = Conv3D(1, (1, 1, 1), activation='sigmoid')(M_concat)
        temporal_attention = temporal_attention[:, 0, :, :, :]

        temporal_attention = Multiply()([cur_frame_conv, temporal_attention])

        print(cur_frame_conv.shape, M_concat.shape, temporal_attention.shape)

        # M_added = tf.keras.layers.Add()([M1_conv, M2_conv, M3_conv, M4_conv])
        # M_sigmoid = Conv2D(M_added.shape[1], (1, 1), activation='sigmoid')(M_added)
        # # temporal_attention = Multiply()([cur_frame_conv, M1_sigmoid])
        # temporal_attention = concatenate([Multiply()([cur_frame_conv, M_sigmoid]), cur_frame_conv], axis=1)

        # print(cur_frame_conv.shape, M_concat.shape, M_sigmoid.shape, temporal_attention.shape)
        print('---------')

        return temporal_attention

    # ------------------------ Model Creation -----------------------------------
    # M stands for Memory, M1 means previous 1 frame from current frame
    cur_frame_input = Input(shape=[3, img_rows, img_cols])
    M1_input = Input(shape=[3, img_rows, img_cols])
    M2_input = Input(shape=[3, img_rows, img_cols])
    M3_input = Input(shape=[3, img_rows, img_cols])
    M4_input = Input(shape=[3, img_rows, img_cols])

    a_encoder = net_block.create_VGG19D_encoder(cur_frame_input)
    block1_conv, block2_conv, block3_conv, block4_conv, block5_conv = a_encoder(cur_frame_input)
    M1_block1_conv, M1_block2_conv, M1_block3_conv, M1_block4_conv, M1_block5_conv = a_encoder(M1_input)
    M2_block1_conv, M2_block2_conv, M2_block3_conv, M2_block4_conv, M2_block5_conv = a_encoder(M2_input)
    M3_block1_conv, M3_block2_conv, M3_block3_conv, M3_block4_conv, M3_block5_conv = a_encoder(M3_input)
    M4_block1_conv, M4_block2_conv, M4_block3_conv, M4_block4_conv, M4_block5_conv = a_encoder(M4_input)

    block1_conv = temporal_attn_block(block1_conv, M1_block1_conv, M2_block1_conv, M3_block1_conv, M4_block1_conv)
    block2_conv = temporal_attn_block(block2_conv, M1_block2_conv, M2_block2_conv, M3_block2_conv, M4_block2_conv)
    block3_conv = temporal_attn_block(block3_conv, M1_block3_conv, M2_block3_conv, M3_block3_conv, M4_block3_conv)
    block4_conv = temporal_attn_block(block4_conv, M1_block4_conv, M2_block4_conv, M3_block4_conv, M4_block4_conv)
    block5_conv = temporal_attn_block(block5_conv, M1_block5_conv, M2_block5_conv, M3_block5_conv, M4_block5_conv)

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

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19D_temporal_attn_v3(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    # M stands for Memory, M1 means previous 1 frame from current frame
    cur_frame_input = Input(shape=[3, img_rows, img_cols])
    M1_input = Input(shape=[3, img_rows, img_cols])
    M2_input = Input(shape=[3, img_rows, img_cols])
    M3_input = Input(shape=[3, img_rows, img_cols])
    M4_input = Input(shape=[3, img_rows, img_cols])

    VGG19D_model = create_VGG19D(cur_frame_input, crop_margin, right_crop, bottom_crop)
    cur_frame_output = VGG19D_model(cur_frame_input)
    M1_output = VGG19D_model(M1_input)
    M2_output = VGG19D_model(M2_input)
    M3_output = VGG19D_model(M3_input)
    M4_output = VGG19D_model(M4_input)

    conv9 = net_block.temporal_attn_block(cur_frame_output, M1_output, M2_output, M3_output, M4_output, 512)

    sigmoid_conv = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[cur_frame_input, M1_input, M2_input, M3_input, M4_input], outputs=[sigmoid_conv])

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19D_temporal_distributed(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):

    def temporal_distributed_block(cur_frame_conv, M1_conv, M2_conv, M3_conv, M4_conv, value_channel_num):
        def create_encoding_block(input_conv, channel_num):
            input_conv = Input(shape=input_conv.shape[1:])
            a_conv = Conv2D(channel_num, (1, 1), activation='relu', padding='same')(input_conv)
            # a_conv = Conv2D(channel_num, (3, 3), activation='relu', padding='same')(a_conv)

            return Model(inputs=input_conv, outputs=a_conv)

        def attention_operator(cur_query, prev_key, prev_value, cur_value):
            key_channel_num = prev_key.shape[1]
            value_channel_num = prev_value.shape[1]
            height, width = cur_query.shape[2:]
            print(cur_query.shape, prev_key.shape, prev_value.shape, cur_value.shape)

            # C x H x W --> C x HW
            cur_query = tf.reshape(cur_query, shape=(-1, key_channel_num, height * width))
            prev_key = tf.reshape(prev_key, shape=(-1, key_channel_num, height * width))
            prev_value = tf.reshape(prev_value, shape=(-1, value_channel_num, height * width))
            print(cur_query.shape, prev_key.shape, prev_value.shape, cur_value.shape)

            # c x hw --> hw x c
            prev_key_transpose = tf.transpose(prev_key, perm=[0, 2, 1])
            print(prev_key_transpose.shape)
            # tf.print(tf.reduce_max(cur_query))
            # tf.print(tf.reduce_max(prev_key_transpose))
            # hw x c and c x hw  --> hw x hw
            Affinity = tf.nn.softmax( tf.matmul(prev_key_transpose, cur_query) / math.sqrt(key_channel_num * height * width))
            print(Affinity.shape)
            # tf.print(tf.reduce_max(Affinity))
            # c x hw and hw x hw --> c x hw
            affinity_prev_value = tf.matmul(prev_value, Affinity)
            print(affinity_prev_value.shape)

            # C x HW --> C x H x W
            affinity_prev_value = tf.reshape(affinity_prev_value, shape=(-1, value_channel_num, height, width))
            print(affinity_prev_value.shape)

            # future_value = Conv2D(value_channel_num, (1, 1), activation='relu')(affinity_prev_value) + cur_value
            future_value = concatenate([affinity_prev_value, cur_value], axis=1)
            print(future_value.shape)

            return future_value

        print('temporal_distributed_block')
        print(cur_frame_conv.shape, M1_conv.shape, M2_conv.shape, M3_conv.shape, M4_conv.shape, value_channel_num)

        key_channel_num = value_channel_num

        cur_frame_value_encoder = create_encoding_block(cur_frame_conv, value_channel_num)
        cur_frame_query_encoder = create_encoding_block(cur_frame_conv, key_channel_num)

        M_key_encoder = create_encoding_block(M1_conv, key_channel_num)
        M_value_encoder = create_encoding_block(M1_conv, value_channel_num)
        M_query_encoder = create_encoding_block(M1_conv, key_channel_num)

        cur_frame_value = cur_frame_value_encoder(cur_frame_conv)
        cur_frame_query = cur_frame_query_encoder(cur_frame_conv)

        M1_key = M_key_encoder(M1_conv)
        M1_value = M_value_encoder(M1_conv)
        M1_query = M_query_encoder(M1_conv)
        M2_key = M_key_encoder(M2_conv)
        M2_value = M_value_encoder(M2_conv)
        M2_query = M_query_encoder(M2_conv)
        M3_key = M_key_encoder(M3_conv)
        M3_value = M_value_encoder(M3_conv)
        M3_query = M_query_encoder(M3_conv)
        M4_key = M_key_encoder(M4_conv)
        M4_value = M_value_encoder(M4_conv)

        M3_value_updated = attention_operator(M3_query, M4_key, M4_value, M3_value)
        M2_value_updated = attention_operator(M2_query, M3_key, M3_value_updated, M2_value)
        M1_value_updated = attention_operator(M1_query, M2_key, M2_value_updated, M1_value)
        cur_frame_value_updated = attention_operator(cur_frame_query, M1_key, M1_value_updated, cur_frame_value)
        print('---------')

        return cur_frame_value_updated

    # ------------------------ Model Creation -----------------------------------
    # M stands for Memory, M1 means previous 1 frame from current frame
    cur_frame_input = Input(shape=[3, img_rows, img_cols])
    M1_input = Input(shape=[3, img_rows, img_cols])
    M2_input = Input(shape=[3, img_rows, img_cols])
    M3_input = Input(shape=[3, img_rows, img_cols])
    M4_input = Input(shape=[3, img_rows, img_cols])

    VGG19D_model = create_VGG19D(cur_frame_input, crop_margin, right_crop, bottom_crop)
    cur_frame_output = VGG19D_model(cur_frame_input)
    M1_output = VGG19D_model(M1_input)
    M2_output = VGG19D_model(M2_input)
    M3_output = VGG19D_model(M3_input)
    M4_output = VGG19D_model(M4_input)

    temporal_combined_conv = temporal_distributed_block(cur_frame_output, M1_output, M2_output, M3_output, M4_output, 64)

    sigmoid_conv = Conv2D(1, (1, 1), activation='sigmoid')(temporal_combined_conv)

    model = Model(inputs=[cur_frame_input, M1_input, M2_input, M3_input, M4_input], outputs=[sigmoid_conv])

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


def VGG19D_temporal_distributed_v2(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):

    def temporal_distributed_block(cur_frame_conv, M1_conv, M2_conv, M3_conv, M4_conv, value_channel_num):
        def create_encoding_block(input_conv, channel_num):
            input_conv = Input(shape=input_conv.shape[1:])
            a_conv = Conv2D(channel_num, (1, 1), activation='relu', padding='same')(input_conv)
            # a_conv = Conv2D(channel_num, (3, 3), activation='relu', padding='same')(a_conv)

            return Model(inputs=input_conv, outputs=a_conv)

        def attention_operator(cur_query, prev_key, prev_value, cur_value):
            key_channel_num = prev_key.shape[1]
            value_channel_num = prev_value.shape[1]
            height, width = cur_query.shape[2:]
            print(cur_query.shape, prev_key.shape, prev_value.shape, cur_value.shape)

            # C x H x W --> C x HW
            cur_query = tf.reshape(cur_query, shape=(-1, key_channel_num, height * width))
            prev_key = tf.reshape(prev_key, shape=(-1, key_channel_num, height * width))
            prev_value = tf.reshape(prev_value, shape=(-1, value_channel_num, height * width))
            print(cur_query.shape, prev_key.shape, prev_value.shape, cur_value.shape)

            # c x hw --> hw x c
            prev_key_transpose = tf.transpose(prev_key, perm=[0, 2, 1])
            print(prev_key_transpose.shape)
            # tf.print(tf.reduce_max(cur_query))
            # tf.print(tf.reduce_max(prev_key_transpose))
            # c x hw and hw x c  --> c x c
            Affinity = tf.nn.softmax( tf.matmul(cur_query, prev_key_transpose) / math.sqrt(key_channel_num * height * width))
            print(Affinity.shape)
            # tf.print(tf.reduce_max(Affinity))
            # c x c and c x hw --> c x hw
            affinity_prev_value = tf.matmul(Affinity, prev_value)
            print(affinity_prev_value.shape)

            # C x HW --> C x H x W
            affinity_prev_value = tf.reshape(affinity_prev_value, shape=(-1, value_channel_num, height, width))
            print(affinity_prev_value.shape)

            # future_value = Conv2D(value_channel_num, (1, 1), activation='relu')(affinity_prev_value) + cur_value
            future_value = concatenate([affinity_prev_value, cur_value], axis=1)
            print(future_value.shape)

            return future_value

        print('temporal_distributed_block')
        print(cur_frame_conv.shape, M1_conv.shape, M2_conv.shape, M3_conv.shape, M4_conv.shape, value_channel_num)

        key_channel_num = value_channel_num

        cur_frame_value_encoder = create_encoding_block(cur_frame_conv, value_channel_num)
        cur_frame_query_encoder = create_encoding_block(cur_frame_conv, key_channel_num)

        M_key_encoder = create_encoding_block(M1_conv, key_channel_num)
        M_value_encoder = create_encoding_block(M1_conv, value_channel_num)
        M_query_encoder = create_encoding_block(M1_conv, key_channel_num)

        cur_frame_value = cur_frame_value_encoder(cur_frame_conv)
        cur_frame_query = cur_frame_query_encoder(cur_frame_conv)

        M1_key = M_key_encoder(M1_conv)
        M1_value = M_value_encoder(M1_conv)
        M1_query = M_query_encoder(M1_conv)
        M2_key = M_key_encoder(M2_conv)
        M2_value = M_value_encoder(M2_conv)
        M2_query = M_query_encoder(M2_conv)
        M3_key = M_key_encoder(M3_conv)
        M3_value = M_value_encoder(M3_conv)
        M3_query = M_query_encoder(M3_conv)
        M4_key = M_key_encoder(M4_conv)
        M4_value = M_value_encoder(M4_conv)

        # M3_value_updated = attention_operator(M3_query, M4_key, M4_value, M3_value)
        # M2_value_updated = attention_operator(M2_query, M3_key, M3_value, M2_value)
        # M1_value_updated = attention_operator(M1_query, M2_key, M2_value, M1_value)
        cur_frame_value_updated = attention_operator(cur_frame_query, M1_key, M1_value, cur_frame_value)
        print('---------')

        return cur_frame_value_updated

    # ------------------------ Model Creation -----------------------------------
    # M stands for Memory, M1 means previous 1 frame from current frame
    cur_frame_input = Input(shape=[3, img_rows, img_cols])
    M1_input = Input(shape=[3, img_rows, img_cols])
    M2_input = Input(shape=[3, img_rows, img_cols])
    M3_input = Input(shape=[3, img_rows, img_cols])
    M4_input = Input(shape=[3, img_rows, img_cols])

    a_encoder = net_block.create_VGG19D_encoder(cur_frame_input)
    block1_conv, block2_conv, block3_conv, block4_conv, block5_conv = a_encoder(cur_frame_input)
    M1_block1_conv, M1_block2_conv, M1_block3_conv, M1_block4_conv, M1_block5_conv = a_encoder(M1_input)
    M2_block1_conv, M2_block2_conv, M2_block3_conv, M2_block4_conv, M2_block5_conv = a_encoder(M2_input)
    M3_block1_conv, M3_block2_conv, M3_block3_conv, M3_block4_conv, M3_block5_conv = a_encoder(M3_input)
    M4_block1_conv, M4_block2_conv, M4_block3_conv, M4_block4_conv, M4_block5_conv = a_encoder(M4_input)

    block1_conv = temporal_distributed_block(block1_conv, M1_block1_conv, M2_block1_conv, M3_block1_conv, M4_block1_conv, 64)
    block2_conv = temporal_distributed_block(block2_conv, M1_block2_conv, M2_block2_conv, M3_block2_conv, M4_block2_conv, 128)
    block3_conv = temporal_distributed_block(block3_conv, M1_block3_conv, M2_block3_conv, M3_block3_conv, M4_block3_conv, 256)
    block4_conv = temporal_distributed_block(block4_conv, M1_block4_conv, M2_block4_conv, M3_block4_conv, M4_block4_conv, 512)
    block5_conv = temporal_distributed_block(block5_conv, M1_block5_conv, M2_block5_conv, M3_block5_conv, M4_block5_conv, 512)

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

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19D_temporal_context_residual(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):
    def context_residual_block(cur_frame_conv, M1_conv, M2_conv, M3_conv, M4_conv):
        print('context_residual_block')

        # subtract features between each pairs of frames
        subtracted1 = Subtract()([cur_frame_conv, M1_conv])
        subtracted2 = Subtract()([M1_conv, M2_conv])
        subtracted3 = Subtract()([M2_conv, M3_conv])
        subtracted4 = Subtract()([M3_conv, M4_conv])

        # concatenate those subtracted features
        # context_residual = tf.keras.layers.Add()([tf.keras.backend.abs(subtracted1),
        #                                           tf.keras.backend.abs(subtracted2),
        #                                           tf.keras.backend.abs(subtracted3),
        #                                           tf.keras.backend.abs(subtracted4)])
        # context_residual = tf.keras.backend.abs(concatenate([subtracted1,subtracted2,subtracted3,subtracted4], axis=1))

        subtracted_concat = concatenate([tf.expand_dims(subtracted1, axis=1),
                                        tf.expand_dims(subtracted2, axis=1),
                                        tf.expand_dims(subtracted3, axis=1),
                                        tf.expand_dims(subtracted4, axis=1)], axis=1)
        subtracted_concat = tf.keras.backend.abs(subtracted_concat)

        # convolve 1x1 to reduce their channels to equal channels of (cur_frame_conv)
        conv_context_residual = Conv3D(1, (1, 1, 1), activation='sigmoid')(subtracted_concat)
        conv_context_residual = conv_context_residual[:,0,:,:,:]
        print(cur_frame_conv.shape, subtracted_concat.shape, conv_context_residual.shape)

        context_attention = Multiply()([cur_frame_conv, conv_context_residual])
        # context_attention = Add()([Multiply()([cur_frame_conv, conv_context_residual]), cur_frame_conv])

        print(context_attention.shape)
        print('---------')

        return context_attention

    # ------------------------ Model Creation -----------------------------------
    # M stands for Memory, M1 means previous 1 frame from current frame
    cur_frame_input = Input(shape=[3, img_rows, img_cols])
    M1_input = Input(shape=[3, img_rows, img_cols])
    M2_input = Input(shape=[3, img_rows, img_cols])
    M3_input = Input(shape=[3, img_rows, img_cols])
    M4_input = Input(shape=[3, img_rows, img_cols])

    VGG19D_model = create_VGG19D(cur_frame_input, crop_margin, right_crop, bottom_crop)
    cur_frame_output = VGG19D_model(cur_frame_input)
    M1_output = VGG19D_model(M1_input)
    M2_output = VGG19D_model(M2_input)
    M3_output = VGG19D_model(M3_input)
    M4_output = VGG19D_model(M4_input)

    conv9 = context_residual_block(cur_frame_output, M1_output, M2_output, M3_output, M4_output)

    sigmoid_conv = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[cur_frame_input, M1_input, M2_input, M3_input, M4_input], outputs=[sigmoid_conv])

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


def VGG19D_temporal_se(img_rows, img_cols, crop_margin, right_crop, bottom_crop, weights_path):

    def temporal_se_block(cur_frame_conv, M1_conv, M2_conv, M3_conv, M4_conv):
        # concatenate previous features
        M_concat = concatenate([tf.expand_dims(M1_conv, axis=1),
                    tf.expand_dims(M2_conv, axis=1),
                    tf.expand_dims(M3_conv, axis=1),
                    tf.expand_dims(M4_conv, axis=1)], axis=1)
        M_concat = tf.keras.backend.abs(M_concat)
        print(M_concat.shape)
        M1_conv = squeeze_and_excitation(M1_conv)


        # convolve 1x1 to reduce their channels to equal channels of (cur_frame_conv)
        conv_context_residual = Conv2D(int(context_residual.shape[1] / 4), (1, 1), activation='relu', padding='same')(
            context_residual)
        conv_context_residual = Conv2D(conv_context_residual.shape[1], (1, 1), activation='sigmoid')(
            conv_context_residual)

        # multiply with (cur_frame_conv) and add (cur_frame_conv)
        context_attention = Add()([Multiply()([cur_frame_conv, conv_context_residual]), cur_frame_conv])

        print(cur_frame_conv.shape, context_residual.shape, conv_context_residual.shape, context_attention.shape)
        print('---------')

        return context_attention

    # ------------------------ Model Creation -----------------------------------
    # M stands for Memory, M1 means previous 1 frame from current frame
    cur_frame_input = Input(shape=[3, img_rows, img_cols])
    M1_input = Input(shape=[3, img_rows, img_cols])
    M2_input = Input(shape=[3, img_rows, img_cols])
    M3_input = Input(shape=[3, img_rows, img_cols])
    M4_input = Input(shape=[3, img_rows, img_cols])

    a_encoder = net_block.create_VGG19D_encoder(cur_frame_input)
    block1_conv, block2_conv, block3_conv, block4_conv, block5_conv = a_encoder(cur_frame_input)
    M1_block1_conv, M1_block2_conv, M1_block3_conv, M1_block4_conv, M1_block5_conv = a_encoder(M1_input)
    M2_block1_conv, M2_block2_conv, M2_block3_conv, M2_block4_conv, M2_block5_conv = a_encoder(M2_input)
    M3_block1_conv, M3_block2_conv, M3_block3_conv, M3_block4_conv, M3_block5_conv = a_encoder(M3_input)
    M4_block1_conv, M4_block2_conv, M4_block3_conv, M4_block4_conv, M4_block5_conv = a_encoder(M4_input)

    block1_conv = temporal_se_block(block1_conv, M1_block1_conv, M2_block1_conv, M3_block1_conv, M4_block1_conv)
    block2_conv = temporal_se_block(block2_conv, M1_block2_conv, M2_block2_conv, M3_block2_conv, M4_block2_conv)
    block3_conv = temporal_se_block(block3_conv, M1_block3_conv, M2_block3_conv, M3_block3_conv, M4_block3_conv)
    block4_conv = temporal_se_block(block4_conv, M1_block4_conv, M2_block4_conv, M3_block4_conv, M4_block4_conv)
    block5_conv = temporal_se_block(block5_conv, M1_block5_conv, M2_block5_conv, M3_block5_conv, M4_block5_conv)

    # ------------------ Decoder -----------------------
    a_decoder = create_decoder(Input(shape=block1_conv.shape[1:]), Input(shape=block2_conv.shape[1:]),
                               Input(shape=block3_conv.shape[1:]), Input(shape=block4_conv.shape[1:]),
                               Input(shape=block5_conv.shape[1:]), crop_margin, bottom_crop, right_crop)
    conv6, conv7, conv8, conv9, crop_conv = a_decoder(block1_conv, block2_conv, block3_conv, block4_conv, block5_conv)

    model = Model(inputs=[cur_frame_input, M1_input, M2_input, M3_input, M4_input], outputs=[crop_conv])

    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


def create_VGG19D(inputs, crop_margin, right_crop, bottom_crop):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    block1_conv = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    block2_conv = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    block3_conv = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv)
    x = Dropout(0.5)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    block4_conv = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv)
    x = Dropout(0.5)(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    block5_conv = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)

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

    if bottom_crop == 0:
        # ((top_crop, bottom_crop), (left_crop, right_crop)) for training
        crop_conv = Cropping2D(cropping=((crop_margin, crop_margin), (crop_margin, crop_margin)))(conv9)
    else:
        # remove reflected portion from the image for prediction
        crop_conv = Cropping2D(cropping=((0, bottom_crop), (0, right_crop)))(conv9)

    model = tf.keras.Model(inputs=inputs,
                           outputs=crop_conv)
    # Load weights.
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

