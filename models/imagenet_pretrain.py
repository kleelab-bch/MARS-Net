'''
Junbong Jang
6/11/2021

Pretrain any models on ImageNet so that the model can be finetuned later on.
'''

import sys
sys.path.append('..')
from UserParams import UserParams

import tensorflow as tf
import glob
import os

from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Activation, Add, Input, concatenate, Conv2D, MaxPooling2D,
                                     AveragePooling2D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose,
                                     BatchNormalization, Dropout, GaussianNoise, Flatten,
                                     GlobalAveragePooling2D, Dense, Flatten, ReLU)
from tensorflow.keras.optimizers import Adam, SGD

from custom_callback import TimeHistory
from deep_neural_net_classifier import *


K.set_image_data_format('channels_last')
print(K.image_data_format())
constants = UserParams('train')
repeat_index = 0
model_name = 'A'
frame = 0
input_size = 224
batch_size = 128

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import numpy as np


def smart_resize(x, size, interpolation='bilinear'):
    # referenced https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/keras/preprocessing/image.py#L50-L144
    """Resize images to a target size without aspect ratio distortion.
    TensorFlow image datasets typically yield images that have each a different
    size. However, these images need to be batched before they can be
    processed by Keras layers. To be batched, images need to share the same height
    and width.
    You could simply do:
    ```python
    size = (200, 200)
    ds = ds.map(lambda img: tf.image.resize(img, size))
    ```
    However, if you do this, you distort the aspect ratio of your images, since
    in general they do not all have the same aspect ratio as `size`. This is
    fine in many cases, but not always (e.g. for GANs this can be a problem).
    Note that passing the argument `preserve_aspect_ratio=True` to `resize`
    will preserve the aspect ratio, but at the cost of no longer respecting the
    provided target size. Because `tf.image.resize` doesn't crop images,
    your output images will still have different sizes.
    This calls for:
    ```python
    size = (200, 200)
    ds = ds.map(lambda img: smart_resize(img, size))
    ```
    Your output images will actually be `(200, 200)`, and will not be distorted.
    Instead, the parts of the image that do not fit within the target size
    get cropped out.
    The resizing process is:
    1. Take the largest centered crop of the image that has the same aspect ratio
    as the target size. For instance, if `size=(200, 200)` and the input image has
    size `(340, 500)`, we take a crop of `(340, 340)` centered along the width.
    2. Resize the cropped image to the target size. In the example above,
    we resize the `(340, 340)` crop to `(200, 200)`.
    Args:
    x: Input image (as a tensor or NumPy array). Must be in format
      `(height, width, channels)`.
    size: Tuple of `(height, width)` integer. Target size.
    interpolation: String, interpolation to use for resizing.
      Defaults to `'bilinear'`. Supports `bilinear`, `nearest`, `bicubic`,
      `area`, `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
    Returns:
    Array with shape `(size[0], size[1], channels)`. If the input image was a
    NumPy array, the output is a NumPy array, and if it was a TF tensor,
    the output is a TF tensor.
    """
    if len(size) != 2:
        raise ValueError('Expected `size` to be a tuple of 2 integers, '
                         'but got: %s' % (size,))
    img = ops.convert_to_tensor_v2_with_dispatch(x)
    if img.shape.rank is not None:
        if img.shape.rank != 3:
            raise ValueError(
              'Expected an image array with shape `(height, width, channels)`, but '
              'got input with incorrect rank, of shape %s' % (img.shape,))

    shape = array_ops.shape(img)
    height, width = shape[0], shape[1]
    target_height, target_width = size

    crop_height = math_ops.cast(
      math_ops.cast(width * target_height, 'float32') / target_width, 'int32')
    crop_width = math_ops.cast(
      math_ops.cast(height * target_width, 'float32') / target_height, 'int32')

    # Set back to input height / width if crop_height / crop_width is not smaller.
    crop_height = math_ops.minimum(height, crop_height)
    crop_width = math_ops.minimum(width, crop_width)

    crop_box_hstart = math_ops.cast(
      math_ops.cast(height - crop_height, 'float32') / 2, 'int32')
    crop_box_wstart = math_ops.cast(
      math_ops.cast(width - crop_width, 'float32') / 2, 'int32')

    crop_box_start = array_ops.stack([crop_box_hstart, crop_box_wstart, 0])
    crop_box_size = array_ops.stack([crop_height, crop_width, -1])

    img = array_ops.slice(img, crop_box_start, crop_box_size)
    img = image_ops.resize_images_v2(
                      images=img,
                      size=size,
                      method=interpolation)

    if isinstance(x, np.ndarray):
        return img.numpy()
    return img


def parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    # resize to size 256 x 256
    image = smart_resize(image, [input_size, input_size], interpolation='bilinear')
    # image = tf.image.per_image_standardization(image)

    return image, label


def preproces_image(image, label):
    image = tf.image.per_image_standardization(image)

    return image, label


def get_filenames_from_directory(directory_path):
    from sklearn.preprocessing import OneHotEncoder
    subdirectory = [name for name in os.listdir(directory_path) if os.path.isdir(directory_path+name)]
    class_i = 0
    total_img_filenames = []
    total_class_list = []

    for a_sub in subdirectory:
        img_filenames = glob.glob(f'{directory_path}/{a_sub}/*.JPEG')
        class_list = [class_i for img_filename in img_filenames]
        class_i = class_i + 1


        total_img_filenames.extend(img_filenames)
        total_class_list.extend(class_list)

    # one hot encode the interger class in the class list since VGG16 has dense layer with 1000 output

    # np_total_class_list = np.asarray(total_class_list).reshape(len(total_class_list), 1)
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded_total_class_list = onehot_encoder.fit_transform(np_total_class_list)

    onehot_encoded_total_class_list = tf.keras.utils.to_categorical(total_class_list, num_classes=1000)
    print('get_filenames_from_directory', len(total_img_filenames), len(set(total_class_list)), onehot_encoded_total_class_list.shape)

    return total_img_filenames, onehot_encoded_total_class_list


if __name__ == "__main__":
    # Data pipeline that reads imagenet data in directory and process them before training
    # referenced https://cs230.stanford.edu/blog/datapipeline/
    with tf.device('/cpu:0'):
        # imagenet_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_data_directory,
        #                                                                             seed=0,
        #                                                                             labels='inferred',
        #                                                                             image_size=(input_size, input_size),
        #                                                                             label_mode='int',
        #                                                                             batch_size=batch_size,
        #                                                                             shuffle=True,
        #                                                                             validation_split=0.05,
        #                                                                             subset='training')

        # imagenet_valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(train_data_directory,
        #                                                                             seed=0,
        #                                                                             labels='inferred',
        #                                                                             image_size=(input_size, input_size),
        #                                                                             label_mode='int',
        #                                                                             batch_size=batch_size,
        #                                                                             shuffle=True,
        #                                                                             validation_split=0.05,
        #                                                                             subset='validation')

        train_data_directory = '/home/qci/imagenet/train/'  # '../assets/imagenet/Data/CLS-LOC/train/'
        filenames, labels = get_filenames_from_directory(train_data_directory)
        DATASET_SIZE = len(filenames)

        full_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        full_dataset = full_dataset.shuffle(len(filenames))

        # split dataset
        train_size = int(0.95 * DATASET_SIZE)
        val_size = int(0.05 * DATASET_SIZE)
        train_dataset = full_dataset.take(train_size)
        valid_dataset = full_dataset.skip(train_size)

        # preprocess every image in the dataset using `map`
        train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
        valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=4)

        # Augment data: random horizontal flipping and randomRGB colour shift

        # batch
        train_dataset = train_dataset.batch(batch_size)
        valid_dataset = valid_dataset.batch(batch_size)

        # prefetch
        train_dataset = train_dataset.prefetch(10)
        valid_dataset = valid_dataset.prefetch(10)
        print('train_dataset', train_dataset, 'valid_dataset', valid_dataset)

    # ---------- Train -----------
    # learning rate decreased by a factor of 10 when the validation set accuracy stopped improving
    # stopped after 370K iterations (74 epochs).
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        epochs = 74
        if constants.strategy_type == 'unet_encoder_classifier':
            model = UNet_encoder_classifier(input_size, input_size, '')
        elif constants.strategy_type == 'VGG16_classifier':
            model = VGG16_classifier(input_size, input_size, '')
        else:
            model = None
        top_1_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top_1_categorical_accuracy")
        top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_categorical_accuracy")

        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=False, name="SGD_momentum"),
                      loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=False, name='categorical_crossentropy')],
                      metrics=[top_1_accuracy, top_5_accuracy])

    # for debugging
    # y_true = np.asarray(list(train_dataset.take(1).as_numpy_iterator())[0][1])
    # y_true = y_true[0]
    # print(y_true, y_true.shape, np.amax(y_true), np.argmax(y_true))
    #
    # a_image = list(train_dataset.take(1).as_numpy_iterator())[0][0]
    # y_pred = model.predict(a_image, batch_size=1)
    # y_pred = y_pred[0]
    # print(y_pred, y_pred.shape, np.amax(y_pred), np.argmax(y_pred))
    #
    # cce = tf.keras.losses.CategoricalCrossentropy()
    # print(cce(y_true, y_pred).numpy())

    # ------------ Sanity check the model ------------
    print(model.summary())
    print('Num of layers: ', len(model.layers))
    plot_model(model, to_file='model_plots/model_round{}_{}_train.png'.format(constants.round_num,
                                                                              constants.strategy_type),
               show_shapes=True, show_layer_names=True, dpi=144)

    # ------------ Fit the Model ------------
    print('Fit Model...', epochs)
    model_checkpoint = ModelCheckpoint(
        'results/model_round{}_{}/model_frame{}_{}_repeat{}.hdf5'.format(constants.round_num, constants.strategy_type,
                                                                         str(frame), model_name,
                                                                         str(repeat_index)),
                                                                        monitor='val_loss', verbose=1,
                                                                        save_best_only=False, save_freq='epoch')

    time_callback = TimeHistory()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5, min_lr=0.0001)
    logdir = 'results/history_round{}_{}/tensorboard_frame{}_{}_repeat{}_{}'.format(constants.round_num,
                                                                                    constants.strategy_type, str(frame),
                                                                                    model_name,
                                                                                    str(repeat_index),
                                                                                    datetime.now().strftime(
                                                                                        "%Y%m%d-%H%M%S"))

    hist = model.fit(train_dataset,
                     epochs=epochs,
                     verbose=1,
                     workers=1,
                     batch_size=batch_size,
                     validation_data=valid_dataset,
                     callbacks=[model_checkpoint, time_callback, reduce_lr, TensorBoard(log_dir=logdir)])

    hist.history['times'] = time_callback.times
    print('Save History...')
    np.save('results/history_round{}_{}/history_frame{}_{}_repeat{}.npy'.format(constants.round_num,
                                                                                constants.strategy_type, str(frame),
                                                                                model_name,
                                                                                str(repeat_index)), hist.history)
    K.clear_session()
