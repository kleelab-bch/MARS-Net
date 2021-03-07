'''
Author: Junbong Jang
Creation Date: 9/21/2020

Helper blocks for deep_neural_net.py

'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Activation, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, Conv2DTranspose,BatchNormalization, Dropout, GaussianNoise, ReLU
import tensorflow.keras.backend as K
import math

from scipy.stats import bernoulli
import copy


def downsample_batch_dropout(input_tensor, filters, size, name='', apply_batchnorm=True, activation='ReLU'):
    if name == '':
        x = Conv2D(filters, size, padding='same', use_bias=False)(input_tensor)
    else:
        x = Conv2D(filters, size, name=name, padding='same')(input_tensor)

    x = DropBlock2D(block_size=7, keep_prob=0.9)(x)

    if apply_batchnorm:
        x = BatchNormalization(axis=1)(x)

    if activation!='ReLU':
        x = Activation(activation)(x)
    else:
        x = ReLU()(x)

    return x


def downsample_batch(input_tensor, filters, size, name='', apply_batchnorm=True, activation='ReLU'):
    if name == '':
        x = Conv2D(filters, size, padding='same', use_bias=False)(input_tensor)
    else:    
        x = Conv2D(filters, size, name=name, padding='same')(input_tensor)

    if apply_batchnorm:
        x = BatchNormalization(axis=1)(x)

    if activation!='ReLU':
        x = Activation(activation)(x)
    else:
        x = ReLU()(x)

    return x


def downsample_instance(input_tensor, filters, size, name='', apply_instancenorm=True):
    import tensorflow_addons as tfa
    if name == '':
        x = Conv2D(filters, size, padding='same', use_bias=False)(input_tensor)
    else:    
        x = Conv2D(filters, size, name=name, padding='same')(input_tensor)

    if apply_instancenorm:
        x = tfa.layers.InstanceNormalization(axis=1)(x)

    x = ReLU()(x)

    return x
    

# referenced from https://github.com/Guzaiwang/CE-Net/blob/master/networks/cenet.py
# 9/26/2020
def DACblock(input_tensor, filters):
    dilate1 = Conv2D(filters, kernel_size=(3,3), dilation_rate=(1,1), padding='same', activation='relu')
    dilate2 = Conv2D(filters, kernel_size=(3,3), dilation_rate=(3,3), padding='same', activation='relu')
    dilate3 = Conv2D(filters, kernel_size=(3,3), dilation_rate=(5,5), padding='same', activation='relu')
    conv1x1 = Conv2D(filters, kernel_size=(1,1), dilation_rate=(1,1), padding='same', activation='relu')

    dilate1_out = dilate1(input_tensor)
    dilate2_out = conv1x1(dilate2(input_tensor))
    dilate3_out = conv1x1(dilate2(dilate1(input_tensor)))
    dilate4_out = conv1x1(dilate3(dilate2(dilate1(input_tensor))))

    out = input_tensor + dilate1_out + dilate2_out + dilate3_out + dilate4_out

    return out
    

def SPPblock(input_tensor):
    # Assume input is 16, 16
    in_channels, in_h, in_w = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
    
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) # 8,8
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(3,3)) # 5,5
    pool3 = MaxPooling2D(pool_size=(5, 5), strides=(5,5)) # 3,3
    pool4 = MaxPooling2D(pool_size=(6, 6), strides=(6,6)) # 2,2
    conv = Conv2D(filters=1, kernel_size=1, padding='valid')

    conv_pool1 = conv(pool1(input_tensor))
    conv_pool2 = conv(pool2(input_tensor))
    conv_pool3 = conv(pool3(input_tensor))
    conv_pool4 = conv(pool4(input_tensor))

    layer1 = UpSamplingUnet(size=(in_h/conv_pool1.shape[2], in_w/conv_pool1.shape[3]), interpolation='bilinear')(conv_pool1)
    layer2 = UpSamplingUnet(size=(in_h/conv_pool2.shape[2], in_w/conv_pool2.shape[3]), interpolation='bilinear')(conv_pool2)
    layer3 = UpSamplingUnet(size=(in_h/conv_pool3.shape[2], in_w/conv_pool3.shape[3]), interpolation='bilinear')(conv_pool3)
    layer4 = UpSamplingUnet(size=(in_h/conv_pool4.shape[2], in_w/conv_pool4.shape[3]), interpolation='bilinear')(conv_pool4)

    tf.print("input_tensor @@@@@@@@@@@@@@@", input_tensor.shape)
    tf.print("layer1 @@@@@@@@@@@@@@@", layer1.shape)
    tf.print("layer2 @@@@@@@@@@@@@@@", layer2.shape)
    tf.print("layer3 @@@@@@@@@@@@@@@", layer3.shape)
    tf.print("layer4 @@@@@@@@@@@@@@@", layer4.shape)

    out = concatenate([layer1, layer2, layer3, layer4, input_tensor], axis=1) 

    return out


def spatial_pyramid_pool(previous_conv):
    """
    https://github.com/peace195/sppnet/blob/master/alexnet_spp.py
    previous_conv: a tensor vector of previous convolution layer
   
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    in_height, in_width = previous_conv.shape[2].value, previous_conv.shape[3].value
    out_pool_size = [8,6,4,2]
    tf.print("previous_conv @@@@@@@@@@@@@@@", previous_conv.shape)

    spp = previous_conv
    for i in range(len(out_pool_size)):
        h_strd = h_size = math.floor(in_height / out_pool_size[i])
        w_strd = w_size = math.floor(in_width / out_pool_size[i])
        pad_h = out_pool_size[i] * h_size - in_height
        pad_w = out_pool_size[i] * w_size - in_width
        new_previous_conv = tf.pad(previous_conv, tf.constant([[0, 0],[0, 0],[0, pad_h],[0, pad_w]], dtype=tf.int32))
        max_pool = tf.nn.max_pool(new_previous_conv,
                                  ksize=[1, 1, h_size, h_size],
                                  strides=[1, 1, h_strd, w_strd],
                                  padding='SAME')

        upsampled = UpSampling2D(size=(in_height // out_pool_size[i], in_width // out_pool_size[i]), interpolation='bilinear')(max_pool)
        tf.print("upsampled @@@@@@@@@@@@@@@", upsampled.shape)

        spp = tf.concat(axis=1, values=[spp, upsampled])

    return spp


def identity_block(X, f, filters, stage, block, is_training):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    # first component
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2a', trainable=is_training)(X)
    X = Activation('relu')(X)

    # second component
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2b', trainable=is_training)(X)
    X = Activation('relu')(X)

    # third component
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2c', trainable=is_training)(X)

    # final step
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s, is_training):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    # first component
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2a', trainable=is_training)(X)
    X = Activation('relu')(X)

    # second component
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2b', trainable=is_training)(X)
    X = Activation('relu')(X)

    # third component
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=1, name=bn_name_base + '2c', trainable=is_training)(X)

    # more step
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=1, name=bn_name_base + '1', trainable=is_training)(X_shortcut)

    # final step
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
    """
    https://github.com/DHZS/tf-dropblock/blob/master/nets/dropblock.py
    """
    def __init__(self, keep_prob, block_size, scale=False, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = scale

    def get_config(self):

        config = super(DropBlock2D, self).get_config().copy()
        config.update({
            'keep_prob': self.keep_prob,
            'block_size': self.block_size,
            'scale': self.scale
        })
        return config


    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(tf.constant(self.scale, dtype=tf.bool),
                             true_fn=lambda: output * tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w, tf.float32), tf.cast(self.h, tf.float32)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask