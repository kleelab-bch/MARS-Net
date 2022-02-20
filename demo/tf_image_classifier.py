'''
Junbogn Jang
6/3/2021

Tensorflow image classification tutorial
Referenced: https://www.tensorflow.org/tutorials/images/classification
'''
import os
# tensorflow import must come after os.environ gpu setting
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf
import pathlib

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Activation, Add, Input, concatenate, Conv2D, MaxPooling2D,
                                     AveragePooling2D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose,
                                     BatchNormalization, Dropout, GaussianNoise, Flatten,
                                     GlobalAveragePooling2D, Dense, Flatten, ReLU)
from tensorflow.keras.utils import plot_model, get_file


def tf_image_classifier():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))

    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(image_count, class_names)
    print(train_ds)
    print(val_ds)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = build_my_model(img_height, img_width, class_names)
    # model = build_default_model(img_height, img_width, class_names)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    print(model.predict(val_ds, batch_size=1, verbose=1))

def build_my_model(img_height, img_width, class_names):
    inputs = Input(shape=[img_height, img_width, 3])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format="channels_last")(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',
                          data_format="channels_last")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format="channels_last")(block1_conv2)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format="channels_last")(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
                          data_format="channels_last")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format="channels_last")(block2_conv2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format="channels_last")(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format="channels_last")(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format="channels_last")(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', data_format="channels_last")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format="channels_last")(block3_conv4)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format="channels_last")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format="channels_last")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format="channels_last")(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', data_format="channels_last")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format="channels_last")(block4_conv4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format="channels_last")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format="channels_last")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format="channels_last")(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', data_format="channels_last")(x)

    x = GlobalAveragePooling2D(data_format="channels_last")(x)
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    output = Dense(len(class_names), activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)

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


def build_default_model(img_height, img_width, class_names):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu', data_format="channels_last"),
        layers.MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"),
        layers.Conv2D(32, 3, padding='same', activation='relu', data_format="channels_last"),
        layers.MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"),
        layers.Conv2D(64, 3, padding='same', activation='relu', data_format="channels_last"),
        layers.MaxPooling2D((2,2), strides=(2,2), data_format="channels_last"),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names))
    ])

    return model


if __name__ == "__main__":
    tf_image_classifier()