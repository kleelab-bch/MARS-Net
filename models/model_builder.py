'''
Author Junbong Jang
Date 6/2/2021

To build model for train.py and predict.py
'''

from deeplabv3 import Deeplabv3
from deep_neural_net_classifier import *
from deep_neural_net_MTL import *
from deep_neural_net import *
from deep_neural_net_3D import *
from deep_neural_net_attn import *
from deep_neural_net_layer import *
import loss

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
if tf.__version__.split('.')[0] == '2':
    import tensorflow_addons as tfa
from sam import SAMModel


def build_model_predict(constants, frame, repeat_index, model_name, image_rows, image_cols, orig_rows, orig_cols):
    weights_path = constants.get_trained_weights_path(str(frame), model_name, str(repeat_index))

    if "VGG19_MTL" in str(constants.strategy_type):
        model = VGG19_MTL(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_classifier_regressor" in str(constants.strategy_type):
        model = VGG19_classifier_regressor(image_rows, image_cols, weights_path=weights_path)
    elif "VGG19_classifier" in str(constants.strategy_type):
        model = VGG19_classifier(image_rows, image_cols, weights_path=weights_path)
    elif "VGG19D_classifier" in str(constants.strategy_type):
        model = VGG19D_classifier(image_rows, image_cols, weights_path=weights_path)
    elif "EFF_B7_classifier" in str(constants.strategy_type):
        model = EFF_B7_classifier(image_rows, image_cols, weights_path=weights_path)
    elif "vit_classifier" in str(constants.strategy_type):
        model = vit_classifier(image_rows, image_cols, 1, weights_path=weights_path)

    # --------------------------------------------------------------------------------

    elif "Res50V2" in str(constants.strategy_type):
        model = ResNet50V2Keras(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "Dense201" in str(constants.strategy_type):
        model = DenseNet201Keras(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "InceptionResV2" in str(constants.strategy_type):
        model = InceptionResV2(image_rows, image_cols, 0, image_cols - orig_cols, image_rows - orig_rows, weights_path=weights_path)
    elif "deeplabv3" in str(constants.strategy_type):
        model = Deeplabv3(input_shape=(image_rows, image_cols, 3), output_shape=(orig_rows, orig_cols))
        model.load_weights(weights_path, by_name=True)

    elif "VGG16_dropout" in str(constants.strategy_type):
        model = VGG16_dropout(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16_batchnorm" in str(constants.strategy_type):
        model = VGG16_batchnorm(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16_instancenorm" in str(constants.strategy_type):
        model = VGG16_instancenorm(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "movie3" in str(constants.strategy_type):
        model = VGG16_movie(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16_dac_input256" in constants.strategy_type:
        model = VGG16_dac(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16_spp_input256" in constants.strategy_type:
        model = VGG16_spp(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG16" in str(constants.strategy_type):
        model = VGG16(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)

    elif "VGG19D_temporal_context_residual" in str(constants.strategy_type):
        model = VGG19D_temporal_context_residual(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19D_temporal_distributed_v2" in str(constants.strategy_type):
        model = VGG19D_temporal_distributed_v2(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19D_temporal_distributed" in str(constants.strategy_type):
        model = VGG19D_temporal_distributed(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19D_temporal_attn_v3" in str(constants.strategy_type):
        model = VGG19D_temporal_attn_v3(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19D_temporal_attn_v2" in str(constants.strategy_type):
        model = VGG19D_temporal_attn_v2(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19D_temporal_attn" in str(constants.strategy_type):
        model = VGG19D_temporal_attn(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19D_crop_first" in str(constants.strategy_type):
        model = VGG19D_crop_first(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19D_se" in str(constants.strategy_type):
        model = VGG19D_se(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout_gelu" in str(constants.strategy_type):
        model = VGG19_dropout_gelu(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout_swish" in str(constants.strategy_type):
        model = VGG19_dropout_swish(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout_dac" in str(constants.strategy_type):
        model = VGG19_dropout_dac(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout_feature_extractor" in str(constants.strategy_type):
        model = VGG19_dropout_feature_extractor(image_rows, image_cols, 0, image_cols - orig_cols, image_rows - orig_rows, weights_path=weights_path)
    elif "VGG19_batchnorm_dropout" in str(constants.strategy_type):
        model = VGG19_batchnorm_dropout(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_batchnorm" in str(constants.strategy_type):
        model = VGG19_batchnorm(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_dropout" in str(constants.strategy_type) or "VGG19D" in str(constants.strategy_type):
        model = VGG19_dropout(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19_freeze" in str(constants.strategy_type):
        model = VGG19_freeze(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path, encoder_weights=None)
    elif "VGG19_imagenet_pretrained" in str(constants.strategy_type):
        model = VGG19_imagenet_pretrained(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "VGG19" in str(constants.strategy_type):
        model = VGG19(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path, encoder_weights=None)

    elif "EFF_B7" in str(constants.strategy_type):
        model = EFF_B7(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)

    elif "unet_3D" in str(constants.strategy_type):
        model = UNet_3D(32, image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "unet_feature_extractor" in str(constants.strategy_type):
        model = UNet_feature_extractor(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "unet_imagenet_pretrained" in str(constants.strategy_type):
        model = UNet_imagenet_pretrained(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "small_unet" in str(constants.strategy_type):
        model = UNet_small(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)
    elif "unet" in str(constants.strategy_type):
        model = UNet(image_rows, image_cols, 0, image_cols-orig_cols, image_rows-orig_rows, weights_path=weights_path)

    return model


def build_model_train(constants, args, frame, model_name):
    pretrained_weights_path = constants.get_pretrained_weights_path(frame, model_name)

    if "VGG19_MTL_auto" in str(constants.strategy_type):
        model = VGG19_MTL_auto(args.input_size, args.input_size, args.cropped_boundary, 0, 0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=None)

    elif "VGG19_MTL" in str(constants.strategy_type):
        model = VGG19_MTL(args.input_size, args.input_size, args.cropped_boundary, 0, 0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy', tf.keras.losses.MeanSquaredLogarithmicError(), tfa.losses.sigmoid_focal_crossentropy],
                      loss_weights={"segmentation":0.75,"regressor":0.01,"classifier":1})

    elif "VGG19_classifier_regressor" in str(constants.strategy_type):
        model = VGG19_classifier_regressor(args.input_size, args.input_size, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=[tf.keras.losses.MeanSquaredLogarithmicError(), tfa.losses.sigmoid_focal_crossentropy],
                      loss_weights={"regressor":0.01,"classifier":1})

    elif "VGG19_classifier_custom_loss" in str(constants.strategy_type):
        model = VGG19_classifier_custom_loss(args.input_size, args.input_size, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=None)

    elif "VGG19_classifier" in str(constants.strategy_type):
        model = VGG19_classifier(args.input_size, args.input_size, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=[tfa.losses.SigmoidFocalCrossEntropy(alpha=0.5)], metrics=['accuracy'])

    elif "VGG19D_classifier" in str(constants.strategy_type):
        model = VGG19D_classifier(args.input_size, args.input_size, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=[tfa.losses.SigmoidFocalCrossEntropy(alpha=0.5)], metrics=['accuracy'])

    elif "EFF_B7_classifier" in str(constants.strategy_type):
        model = EFF_B7_classifier(args.input_size, args.input_size, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=[tfa.losses.SigmoidFocalCrossEntropy(alpha=0.5)], metrics=['accuracy'])

    elif "vit_classifier" in str(constants.strategy_type):
        model = vit_classifier(args.input_size, args.input_size, 1, weights_path=pretrained_weights_path)
        # model = SAMModel(model)
        model.compile(optimizer=Adam(lr=1e-5), loss=[tfa.losses.SigmoidFocalCrossEntropy(alpha=0.5)], metrics=['accuracy'])

    # --------------------------------------------------------------------------------

    elif "Res50V2" in str(constants.strategy_type):
        model = ResNet50V2Keras(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "InceptionResV2" in str(constants.strategy_type):
        model = InceptionResV2(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                               weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "Dense201" in str(constants.strategy_type):
        model = DenseNet201Keras(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                 weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "deeplabv3" in str(constants.strategy_type):
        model = Deeplabv3(input_shape=(args.input_size, args.input_size, 3), output_shape=(68, 68), right_crop=0,
                          bottom_crop=0)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_dropout" in str(constants.strategy_type):
        model = VGG16_dropout(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_batchnorm" in str(constants.strategy_type):
        model = VGG16_batchnorm(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_instancenorm" in str(constants.strategy_type):
        model = VGG16_instancenorm(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                   weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_movie3" in str(constants.strategy_type):
        model = VGG16_movie(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                            weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=loss.temporal_cross_entropy, metrics=[loss.dice_coef])

    elif "VGG16_dice" in str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=[loss.dice_coef], metrics=['binary_crossentropy'])

    elif "VGG16_l2" in str(constants.strategy_type):
        model = VGG16_l2(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                         weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_dac_input256" in constants.strategy_type:
        model = VGG16_dac(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                          weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_spp_input256" in constants.strategy_type:
        model = VGG16_spp(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                          weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_no_pretrain" in str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path, encoder_weights=None)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16" in str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_crop_first" in str(constants.strategy_type):
        model = VGG19D_crop_first(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_se" in str(constants.strategy_type):
        model = VGG19D_se(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_temporal_se" in str(constants.strategy_type):
        model = VGG19D_temporal_se(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_temporal_distributed_v2" in str(constants.strategy_type):
        model = VGG19D_temporal_distributed_v2(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_temporal_distributed" in str(constants.strategy_type):
        model = VGG19D_temporal_distributed(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_temporal_context_residual" in str(constants.strategy_type):
        model = VGG19D_temporal_context_residual(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_temporal_attn_v3" in str(constants.strategy_type):
        model = VGG19D_temporal_attn_v3(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_temporal_attn_v2" in str(constants.strategy_type):
        model = VGG19D_temporal_attn_v2(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19D_temporal_attn" in str(constants.strategy_type):
        model = VGG19D_temporal_attn(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_dac_input256" in str(constants.strategy_type):
        model = VGG19_dropout_dac(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                  weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_feature_extractor" in str(constants.strategy_type):
        model = VGG19_dropout_feature_extractor(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                                weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy', loss.zero_loss],
                      metrics=[loss.dice_coef, loss.zero_loss])

    elif "VGG19_batchnorm_dropout" in str(constants.strategy_type):
        model = VGG19_batchnorm_dropout(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                        weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_gelu" in str(constants.strategy_type):
        model = VGG19_dropout_gelu(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                   weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_swish" in str(constants.strategy_type):
        model = VGG19_dropout_swish(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                    weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout" in str(constants.strategy_type) or "VGG19D" in str(constants.strategy_type):
        model = VGG19_dropout(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                              weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_batchnorm" in str(constants.strategy_type):
        model = VGG19_batchnorm(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_no_pretrain_freeze" in str(constants.strategy_type):
        model = VGG19_freeze(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path, encoder_weights=None)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_no_pretrain" in str(constants.strategy_type):
        model = VGG19(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path, encoder_weights=None)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_imagenet_pretrained" in str(constants.strategy_type):
        model = VGG19_imagenet_pretrained(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_freeze" in str(constants.strategy_type):
        model = VGG19_freeze(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19" in str(constants.strategy_type):
        model = VGG19(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                      weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "EFF_B7" in str(constants.strategy_type):
        model = EFF_B7(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                       weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "unet_3D" in str(constants.strategy_type):
        model = UNet_3D(args.input_depth, args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                     weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "unet_feature_extractor" in str(constants.strategy_type):
        model = UNet_feature_extractor(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                                       weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy', loss.zero_loss],
                      metrics=[loss.dice_coef, loss.zero_loss])

    elif "small_unet" in str(constants.strategy_type):
        model = UNet_small(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                     weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "unet_imagenet_pretrained" in str(constants.strategy_type):
        model = UNet_imagenet_pretrained(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                     weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "unet" in str(constants.strategy_type):
        model = UNet(args.input_size, args.input_size, args.cropped_boundary, 0, 0,
                     weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    return model