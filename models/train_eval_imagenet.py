'''
Junbong Jang
6/11/2021

Pretrain any models on ImageNet so that the model can be finetuned later on.
'''

import sys
sys.path.append('..')
from UserParams import UserParams

import glob
import os
import pandas as pd
from datetime import datetime

import tensorflow as tf
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
from imagenet_classid_to_label import *

K.set_image_data_format('channels_last')
print(K.image_data_format())
constants = UserParams('imagenet')
repeat_index = 0
model_name = 'A'
frame = 0
orig_input_size = 256
crop_input_size = 224
batch_size = 256

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

    shape = array_ops.shape(img)  # e.g. 375, 500, 3
    height, width = shape[0], shape[1]
    target_height, target_width = size  # e.g. [224, 224]

    crop_height = math_ops.cast(
      math_ops.cast(width * target_height, 'float32') / target_width, 'int32')
    crop_width = math_ops.cast(
      math_ops.cast(height * target_width, 'float32') / target_height, 'int32')

    # e.g. crop height and crop width is 500 and 375 respectively
    # Set back to input height / width if crop_height / crop_width is not smaller.
    crop_height = math_ops.minimum(height, crop_height)
    crop_width = math_ops.minimum(width, crop_width)
    # e.g. crop height and crop width is 375 and 375 respectively

    crop_box_hstart = math_ops.cast(
      math_ops.cast(height - crop_height, 'float32') / 2, 'int32')
    crop_box_wstart = math_ops.cast(
      math_ops.cast(width - crop_width, 'float32') / 2, 'int32')

    crop_box_start = array_ops.stack([crop_box_hstart, crop_box_wstart, 0])  # e.g. [0 62 0]
    crop_box_size = array_ops.stack([crop_height, crop_width, -1])  # e.g. [375 375 -1]

    img = array_ops.slice(img, crop_box_start, crop_box_size)
    img = image_ops.resize_images_v2(
                      images=img,
                      size=size,
                      method=interpolation)

    if isinstance(x, np.ndarray):
        return img.numpy()
    return img


def parse_train_input(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    # resize to size 256 x 256
    image = smart_resize(image, [orig_input_size, orig_input_size], interpolation='bilinear')

    return image, label


def preprocess_train_input(image, label):
    # crop patch of size 224 x 224 from the size 256 x 256
    image = tf.image.random_crop(value=image, size=(crop_input_size, crop_input_size, 3))

    # Augment by random horizontal flipping and randomRGB colour shift
    # reference https://www.codestudyblog.com/cnb/0319190238.html
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=25.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    image = tf.image.per_image_standardization(image)

    return image, label


def parse_prediction_input(filename):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    image = smart_resize(image, [crop_input_size, crop_input_size], interpolation='bilinear')
    # tf.print(tf.math.reduce_max(image))
    # image /= 127.5
    # image -= 1.
    # tf.print(tf.math.reduce_max(image), '---------------')

    # image = tf.image.per_image_standardization(image)

    return image


def flip_input_image(image):
    return tf.image.flip_left_right(image)


def get_filenames_from_directory(directory_path):
    subdirectory = [name for name in os.listdir(directory_path) if os.path.isdir(directory_path+name)]
    class_i = 0
    total_img_filenames = []
    total_labels = []

    for a_sub in subdirectory:
        img_filenames = glob.glob(f'{directory_path}/{a_sub}/*.JPEG')
        class_list = [class_i for img_filename in img_filenames]
        class_i = class_i + 1

        total_img_filenames.extend(img_filenames)
        total_labels.extend(class_list)

    # one hot encode the interger class in the class list since VGG16 has dense layer with 1000 output
    # onehot_encoded_total_labels = tf.keras.utils.to_categorical(total_labels, num_classes=1000)
    print('get_filenames_from_directory', len(total_img_filenames), len(set(total_labels)))

    return total_img_filenames, total_labels

# -----------------------------------------------------------

def get_model(input_size, weights_path, strategy_type):
    if 'unet_encoder_classifier' in strategy_type:
        model = UNet_encoder_classifier(input_size, input_size, weights_path)
    elif strategy_type == 'VGG16_imagenet_classifier':
        model = VGG16_imagenet_classifier(input_size, input_size, weights_path)
    elif strategy_type == 'VGG19_imagenet_classifier':
        model = VGG19_imagenet_classifier(input_size, input_size, weights_path)
    elif strategy_type == 'vit_imagenet_classifier':
        model = vit_classifier(input_size, input_size, 1000, weights_path)
    else:
        model = None

    return model


def train():
    # Data pipeline that reads imagenet data in directory and process them before training
    # referenced https://cs230.stanford.edu/blog/datapipeline/
    with tf.device('/cpu:0'):
        train_data_directory = '/home/qci/imagenet/train/'  # '../assets/imagenet/Data/CLS-LOC/train/'
        filenames, labels = get_filenames_from_directory(train_data_directory)
        DATASET_SIZE = len(filenames)

        full_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        full_dataset = full_dataset.shuffle(len(filenames))

        # split dataset
        train_size_ratio = 0.99
        train_size = int(train_size_ratio * DATASET_SIZE)
        train_dataset = full_dataset.take(train_size)
        valid_dataset = full_dataset.skip(train_size)

        # Load data
        train_dataset = train_dataset.map(parse_train_input, num_parallel_calls=4)
        valid_dataset = valid_dataset.map(parse_train_input, num_parallel_calls=4)

        # Preprocess and Augment data
        train_dataset = train_dataset.map(preprocess_train_input, num_parallel_calls=4)
        valid_dataset = valid_dataset.map(preprocess_train_input, num_parallel_calls=4)

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
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    epochs = 74

    model = get_model(crop_input_size, '', constants.strategy_type)

    top_1_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top_1_categorical_accuracy")
    top_5_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_categorical_accuracy")

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=False, name="SGD_momentum"),
                  loss=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, name='categorical_crossentropy')],
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


def evaluate():
    # ------------------- Get validation dataset ----------------------
    # get validation images
    img_filenames = glob.glob(f'/home/qci/imagenet/val/*.JPEG')
    # img_filenames = img_filenames[:1000]

    # get validation labels from csv
    validation_label_path = '/home/qci/imagenet/LOC_val_solution.csv'
    label_df = pd.read_csv(validation_label_path, index_col=0)
    label_df['PredictionString'] = label_df['PredictionString'].apply(lambda label_string: label_string.split(' ')[0])

    # convert training dataset folder names sorted in increasing order into numbers 0~999
    directory_path = '/home/qci/imagenet/train/'
    subdirectory_list = [name for name in os.listdir(directory_path) if os.path.isdir(directory_path+name)]
    # subdirectory_list.sort()

    labelname_to_number = {}
    for i, label_name in enumerate(subdirectory_list):
        labelname_to_number[label_name] = i

    # convert validation label into numbers 0~999 based on conversion above
    label_df['PredictionString'] = label_df['PredictionString'].apply(lambda labelname: labelname_to_number[labelname])
    print('label_df', label_df)
    print('img_filenames', len(img_filenames))
    print('subdirectory_list', len(subdirectory_list))

    full_dataset = tf.data.Dataset.from_tensor_slices(img_filenames)

    # -------------- Parse images in different ways --------------------
    orig_dataset = full_dataset.map(parse_prediction_input, num_parallel_calls=4)
    orig_dataset = orig_dataset.batch(1)
    orig_dataset = orig_dataset.prefetch(1000)

    fliped_dataset = full_dataset.map(parse_prediction_input, num_parallel_calls=4)
    fliped_dataset = full_dataset.map(fliped_dataset, num_parallel_calls=4)
    fliped_dataset = fliped_dataset.map(flip_input_image, num_parallel_calls=4)
    fliped_dataset = fliped_dataset.batch(1)
    fliped_dataset = fliped_dataset.prefetch(1000)

    # ------------------- Load trained model ---------------------
    weights_path = constants.get_trained_weights_path(str(frame), model_name, str(repeat_index))

    model = get_model(crop_input_size, weights_path, constants.strategy_type)

    # ------------------- Predict ----------------------
    # combine multiple prediction results from multiple scales
    # referenced ensemble model prediction https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/
    y_pred_orig = model.predict(orig_dataset, batch_size=1, verbose=1)
    y_pred_fliped = model.predict(fliped_dataset, batch_size=1, verbose=1)

    y_pred = (y_pred_orig + y_pred_fliped) / 2
    # ------------------- evaluate using Top1 and Top5 accuracy -------------------
    y_true = np.zeros(shape=(len(img_filenames)))
    print(y_true.shape, y_pred.shape)
    for i, img_filename in enumerate(img_filenames):
        img_filename = img_filename.split('/')[-1].replace('.JPEG', '') # remove path from filename
        y_true[i] = label_df.loc[img_filename].item()
        # print(img_filename, ': ', int(y_true[i]), np.argmax(y_pred[i]))
        # print(img_filename, ': ', imagenet_classid_to_label_string[int(y_true[i])], ' |  ', imagenet_classid_to_label_string[np.argmax(y_pred[i])])

    m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)  # y_true is sparse, y_pred is dense
    m.update_state(y_true, y_pred)  # e.g. y_true = [2, 1], y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    print(m.result().numpy())

    m = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    m.update_state(y_true, y_pred)
    print(m.result().numpy())

    # from sklearn.metrics import top_k_accuracy_score
    # print(top_k_accuracy_score(y_true, y_pred, k=1), top_k_accuracy_score(y_true, y_pred, k=5))


if __name__ == "__main__":
    train()
    # evaluate()
