"""
Author: Junbong Jang
Date: 6/9/2020

Store functions that define deep learning classifiers
"""
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import ResNet50, ResNet50V2, DenseNet201, InceptionResNetV2
from tensorflow.keras.layers import (Layer, Activation, Add, Input, concatenate, Conv2D, MaxPooling2D,
AveragePooling2D, ZeroPadding2D, UpSampling2D, Cropping2D, Conv2DTranspose,BatchNormalization, Dropout, GaussianNoise,
GlobalAveragePooling2D, Dense, Flatten, ReLU)
from tensorflow.keras import layers
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
    output = Dense(1, activation='sigmoid', name='classifier')(x)

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


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        ) # patches.shape = (None, 14, 14, 768)
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims]) # patches.shape = (None, None, 768)

        return patches

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size
        })
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim':self.projection_dim,
            'projection': self.projection,
            'position_embedding': self.position_embedding
        })
        return config


@log_function_call
def vit_classifier(img_rows, img_cols, num_classes, weights_path):
    image_size = 224  # resize input images to this size
    patch_size = 16  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 768
    num_heads = 12
    transformer_units = [
        projection_dim*2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 12
    mlp_head_units = [2048,1024]  # Size of the dense layers of the final classifier

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x
    # 372.989 Million params --> 680.604 Million params
    # ------- Data augmentation ------------
    #   layers.experimental.preprocessing.Normalization(),
    data_augmentation = Sequential(
        [
            layers.experimental.preprocessing.Resizing(image_size, image_size),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(factor=0.02),
            layers.experimental.preprocessing.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    # Compute the mean and the variance of the training data for normalization.
    # data_augmentation.layers[0].adapt(x_train)

    inputs = layers.Input(shape=[img_rows, img_cols, 3])
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim//num_heads, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    # features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    features = representation

    if num_classes == 1:
        final_output = layers.Dense(num_classes, activation='sigmoid', name='top_dense_sigmoid')(features)
    else:
        final_output = layers.Dense(num_classes, activation='softmax', name='top_dense_softmax')(features)

    model = Model(inputs=inputs, outputs=final_output)

    # Load weights.
    if weights_path != '':
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
def VGG16_imagenet_classifier(img_rows, img_cols, weights_path):
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

    # Load weights.
    if weights_path != '':
        model.load_weights(weights_path, by_name=True)

    return model


@log_function_call
def VGG19_imagenet_classifier(img_rows, img_cols, weights_path):
    inputs = Input(shape=[img_rows, img_cols, 3])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=l2(0.0005))(inputs)
    block1_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_conv2)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=l2(0.0005))(x)
    block2_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_conv2)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=l2(0.0005))(x)
    block3_conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_conv4)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=l2(0.0005))(x)
    block4_conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', kernel_regularizer=l2(0.0005))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_conv4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=l2(0.0005))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4', kernel_regularizer=l2(0.0005))(x)
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
