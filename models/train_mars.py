import sys
sys.path.append('..')
import numpy as np
import time
import os.path
import gc
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

from deeplabv3 import Deeplabv3
from deep_neural_net import *
import loss # tensorflow import must come after os.environ gpu setting
from debug_utils import *
from UserParams import UserParams
from custom_callback import TimeHistory


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def concatenate_split_crops(constants, concatenate_dataset, model_name, dataset_name, frame, repeat_index):
    print('concatenate_split_crops: ', model_name, dataset_name, frame)
    for split_index in range(constants.crop_split_constant):
        print('------------split: {}------------'.format(split_index))
        root_path, save_suffix = constants.get_crop_path(model_name, dataset_name, str(frame), str(split_index), str(repeat_index))
        temp_data = np.load(root_path + save_suffix)

        temp_img = temp_data['arr_0']
        temp_mask = temp_data['arr_1']

        if concatenate_dataset == {}:
            concatenate_dataset['arr_0'] = temp_img
            concatenate_dataset['arr_1'] = temp_mask
        else:
            concatenate_dataset['arr_0'] = np.concatenate((concatenate_dataset['arr_0'], temp_img), axis=0)
            concatenate_dataset['arr_1'] = np.concatenate((concatenate_dataset['arr_1'], temp_mask), axis=0)
        print(concatenate_dataset['arr_0'].shape)
    '''
    split_first_patches_num = 0
    #split_last_data = np.load('../crop/crop_{}_strat{}/'.format(constants.teacher_student, constants.strategy_type) + dataset_name + '_' + str(frame) + '_split' + str(constants.crop_split_constant-1) + '_train_mask.npz')
    split_last_data = np.load('../crop/crop_results/crop_round{}_{}/'.format(constants.round_num, constants.strategy_type) + dataset_name + '_' + str(frame) + '_split' + str(constants.crop_split_constant-1) + '_train_mask.npz')
    split_last_img = split_last_data['arr_0']
    split_last_mask = split_last_data['arr_1'].astype(np.uint8)
    
    for split_index in range(constants.crop_split_constant):
        print('------------split: {}------------'.format(split_index))
        if split_index < constants.crop_split_constant - 1:
            #temp_data = np.load('../crop/crop_{}_strat{}/'.format(constants.teacher_student, constants.strategy_type) + dataset_name + '_' + str(frame) + '_split' + str(split_index) + '_train_mask.npz')
            temp_data = np.load('../crop/crop_results/crop_round{}_{}/'.format(constants.round_num, constants.strategy_type) + dataset_name + '_' + str(frame) + '_split' + str(split_index) + '_train_mask.npz')
            
            temp_img = temp_data['arr_0']
            temp_mask = temp_data['arr_1'].astype(np.uint8)
            split_first_patches_num = temp_img.shape[0]
            
            if split_index == 0 and dataset_index == 0:
                total_patches_num = (temp_img.shape[0]*(constants.crop_split_constant-1)+split_last_data['arr_0'].shape[0])*len(constants.dataset_names)
                print(total_patches_num)
                concatenate_dataset['arr_0'] = np.empty([total_patches_num, 3, 128, 128], dtype=np.float32)
                concatenate_dataset['arr_1'] = np.empty([total_patches_num, 1, 68, 68], dtype=np.uint8)    
            
            previous_offset = dataset_index*((constants.crop_split_constant-1)*split_first_patches_num+split_last_data['arr_0'].shape[0])
            lower_bound = previous_offset + split_index*temp_img.shape[0]
            upper_bound = lower_bound + temp_img.shape[0]
            print(lower_bound, upper_bound)
            print(concatenate_dataset)
            print(concatenate_dataset['arr_0'][lower_bound:upper_bound,:,:,:].shape)
            print(temp_img.shape)
            
            concatenate_dataset['arr_0'][lower_bound:upper_bound,:,:,:] = temp_img
            concatenate_dataset['arr_1'][lower_bound:upper_bound,:,:,:] = temp_mask
         
        else:
            previous_offset = dataset_index*((constants.crop_split_constant-1)*split_first_patches_num+split_last_data['arr_0'].shape[0])
            lower_bound = previous_offset + split_index*split_first_patches_num
            upper_bound = lower_bound + split_last_data['arr_0'].shape[0]
            print(lower_bound, upper_bound)
            print(concatenate_dataset)
            concatenate_dataset['arr_0'][lower_bound:upper_bound,:,:,:] = split_last_img
            concatenate_dataset['arr_1'][lower_bound:upper_bound,:,:,:] = split_last_mask
    '''

    return concatenate_dataset


def get_concatenate_one_movie(constants, model_index, frame, repeat_index):
    print('get_concatenate_one_movie')
    model_name = constants.model_names[model_index]
    return concatenate_split_crops({}, model_name, constants.dataset_names[model_index], frame, repeat_index)


# concatenate all datasets except for the dataset index matching current model index
def get_concatenate_dataset(constants, model_index, frame, repeat_index, leave_one_movie):
    concatenate_dataset = {}
    dataset_list = []
    model_name = constants.model_names[model_index]
    for dataset_index, dataset_name in enumerate(constants.dataset_names):
        if leave_one_movie:
            if model_index != dataset_index:
                dataset_list.append(dataset_name)
                concatenate_dataset = concatenate_split_crops(constants, concatenate_dataset, model_name, dataset_name, frame, repeat_index)
            else:
                print()
                print('omitted dataset:', dataset_name)
                print()
        else:
            dataset_list.append(dataset_name)
            concatenate_dataset = concatenate_split_crops(constants, concatenate_dataset, model_name, dataset_name, frame, repeat_index)

    print('get_concatenate_dataset')
    print(dataset_list)
    return concatenate_dataset


def get_training_dataset(constants, model_index, frame, repeat_index):
    if constants.round_num == 1:
        if 'overfit' in str(constants.strategy_type):
            training_dataset = get_concatenate_one_movie(constants, model_index, frame, repeat_index)
        elif 'one_generalist' in str(constants.strategy_type):
            training_dataset = get_concatenate_dataset(constants, model_index, frame, repeat_index, leave_one_movie=False)
        else:
            training_dataset = get_concatenate_dataset(constants, model_index, frame, repeat_index, leave_one_movie=True)

    elif constants.round_num >= 2: # Self Training
        if constants.strategy_type == 2 or constants.strategy_type == 2.5 or constants.strategy_type == 4.5 or \
                constants.strategy_type == "paxillin_TIRF_normalize_2.5" or constants.strategy_type == "cryptic":
            if constants.round_num == 3:
                training_dataset = get_concatenate_one_movie(constants, model_index, frame, repeat_index)
            else:
                training_dataset = get_concatenate_dataset(constants, model_index, frame, repeat_index, leave_one_movie=False)

        elif constants.strategy_type == 3 or constants.strategy_type == 4:
            training_dataset = get_concatenate_one_movie(constants, model_index, frame, repeat_index)

        elif constants.strategy_type == 5 or constants.strategy_type == 6 or constants.strategy_type == 'movie3' or constants.strategy_type == 'movie3_loss' or constants.strategy_type == 'movie3_proc':
            training_dataset = get_concatenate_dataset(constants, model_index, frame, repeat_index, leave_one_movie=False)

    return training_dataset


def train_model(constants, model_index, frame, repeat_index):
    print(constants.model_names[model_index], ' frame:', frame, ' round_num:', constants.round_num, ' repeat_index:', repeat_index)

    args = constants.get_args()
    training_dataset = get_training_dataset(constants, model_index, frame, repeat_index)
    comb_train = training_dataset['arr_0']
    comb_mask = training_dataset['arr_1']

    # ------------ process dataset ----------------
    comb_train, comb_mask = unison_shuffled_copies(comb_train, comb_mask)

    print(np.mean(comb_train), np.std(comb_train), np.ptp(comb_train))
    print(comb_train.dtype, comb_train.shape)
    print(comb_mask.dtype, comb_mask.shape)
    print('----------')

    # ------------------- Model Creation ---------------------------
    # Set Model Hyper Parameters
    pretrained_weights_path = constants.get_pretrained_weights_path(frame, constants.model_names[model_index])

    print('Load Model...')
    if "Res50V2" == str(constants.strategy_type):
        model = ResNet50V2Keras(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "InceptionResV2" == str(constants.strategy_type):
        model = InceptionResV2(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "Dense201" == str(constants.strategy_type):
        model = DenseNet201Keras(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "deeplabv3" == str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        comb_train = np.moveaxis(comb_train, 1, -1)  # first channel to last channel
        comb_mask = np.moveaxis(comb_mask, 1, -1)
        print(comb_train.dtype, comb_train.shape)
        print(comb_mask.dtype, comb_mask.shape)

        model = Deeplabv3(input_shape=(args.input_size, args.input_size, 3), output_shape=(68, 68), right_crop = 0, bottom_crop = 0)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_dropout" == str(constants.strategy_type):
        model = VGG16_dropout(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_batchnorm" == str(constants.strategy_type):
        model = VGG16_batchnorm(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_instancenorm" == str(constants.strategy_type):
        model = VGG16_instancenorm(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_movie3" == str(constants.strategy_type):
        model = VGG16_movie(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=loss.temporal_cross_entropy, metrics=[loss.dice_coef])

    elif "VGG16_dice" == str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=[loss.dice_coef], metrics=['binary_crossentropy'])

    elif "VGG16_l2" == str(constants.strategy_type):
        model = VGG16_l2(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_dac_input256" == constants.strategy_type:
        model = VGG16_dac(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_spp_input256" == constants.strategy_type:
        model = VGG16_spp(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16_no_pretrain" == str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path, encoder_weights = None)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG16" in str(constants.strategy_type):
        model = VGG16(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_dac_input256" == str(constants.strategy_type):
        model = VGG19_dropout_dac(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout_feature_extractor" in str(constants.strategy_type):
        model = VGG19_dropout_feature_extractor(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy', loss.zero_loss], metrics=[loss.dice_coef, loss.zero_loss])

    elif "VGG19_batchnorm_dropout" in str(constants.strategy_type):
        model = VGG19_batchnorm_dropout(args.input_size, args.input_size, args.cropped_boundary, 0, 0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_dropout" in str(constants.strategy_type):
        model = VGG19_dropout(args.input_size, args.input_size, args.cropped_boundary, 0, 0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_batchnorm" == str(constants.strategy_type):
        model = VGG19_batchnorm(args.input_size, args.input_size, args.cropped_boundary, 0, 0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_no_pretrain" == str(constants.strategy_type):
        model = VGG19(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path, encoder_weights = None)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19_imagenet_pretrained" in str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        comb_train = np.moveaxis(comb_train, 1, -1)  # first channel to last channel
        comb_mask = np.moveaxis(comb_mask, 1, -1)

        model = VGG19_imagenet_pretrained(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "VGG19" in str(constants.strategy_type):
        model = VGG19(args.input_size, args.input_size, args.cropped_boundary, 0, 0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "EFF_B7" == str(constants.strategy_type) or "EFF_B7_no_preprocessing" == str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        comb_train = np.moveaxis(comb_train, 1, -1)  # first channel to last channel
        comb_mask = np.moveaxis(comb_mask, 1, -1)
        print(comb_train.dtype, comb_train.shape)
        print(comb_mask.dtype, comb_mask.shape)

        model = EFF_B7(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "unet_feature_extractor" in str(constants.strategy_type):
        model = UNet_feature_extractor(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy', loss.zero_loss], metrics=[loss.dice_coef, loss.zero_loss])

    elif "unet_imagenet_pretrained" in str(constants.strategy_type):
        K.set_image_data_format('channels_last')
        comb_train = np.moveaxis(comb_train, 1, -1)  # first channel to last channel
        comb_mask = np.moveaxis(comb_mask, 1, -1)

        model = UNet_imagenet_pretrained(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])

    elif "unet" in str(constants.strategy_type):
        model = UNet(args.input_size, args.input_size, args.cropped_boundary,0,0, weights_path=pretrained_weights_path)
        model.compile(optimizer=Adam(lr=1e-5), loss=['binary_crossentropy'], metrics=[loss.dice_coef])


    # ------------ Sanity Check the Model ------------
    print(model.summary())
    print('Num of layers: ', len(model.layers))
    # print('FLOPS: ', get_flops())  # run this after model compilation
    # check_loaded_weights(constants)
    if repeat_index == 0:
        plot_model(model, to_file='model_plots/model_round{}_{}_train.png'.format(constants.round_num, constants.strategy_type), show_shapes=True, show_layer_names=True, dpi=144)

    # ------------ Fit the Model ------------
    print('Fit Model...', args.patience)
    earlyStopping = EarlyStopping(monitor='val_loss', patience = args.patience, verbose=0, mode='auto')
    model_checkpoint = ModelCheckpoint('results/model_round{}_{}/model_frame{}_{}_repeat{}.hdf5'.format(constants.round_num, constants.strategy_type, str(frame), constants.model_names[model_index], str(repeat_index)),
                                       monitor='val_loss', save_best_only=True)
    time_callback = TimeHistory()

    if "feature_extractor" in str(constants.strategy_type):
        hist = model.fit(comb_train, [comb_mask,[]], batch_size = args.train_batch_size, epochs = args.epochs, validation_split = args.validation_split,
                         verbose=1, shuffle=True, callbacks=[model_checkpoint, earlyStopping, time_callback])
    else:
        x_train, x_val, y_train, y_val = train_test_split(comb_train, comb_mask, shuffle=True, test_size=0.2, random_state=repeat_index)
        hist = model.fit(x_train, y_train, batch_size = args.train_batch_size, epochs = args.epochs,
                         validation_data=(x_val, y_val), verbose=1, shuffle=True,
                         callbacks=[model_checkpoint, earlyStopping, time_callback])

    # ------------ Save the History ------------
    hist.history['times'] = time_callback.times
    print('Save History...')
    np.save('results/history_round{}_{}/history_frame{}_{}_repeat{}.npy'.format(constants.round_num, constants.strategy_type, str(frame), constants.model_names[model_index], str(repeat_index)), hist.history)
    K.clear_session()

    return


if __name__ == "__main__":
    K.set_image_data_format('channels_first')
    print(K.image_data_format())
    constants = UserParams('train')
    print('{}_{}'.format(constants.round_num, constants.strategy_type))

    print(os.path.exists('results/history_round{}_{}'.format(constants.round_num, constants.strategy_type)))
    if not os.path.exists('results/history_round{}_{}'.format(constants.round_num, constants.strategy_type)):
        os.makedirs('results/history_round{}_{}'.format(constants.round_num, constants.strategy_type))
    if not os.path.exists('results/model_round{}_{}'.format(constants.round_num, constants.strategy_type)):
        os.makedirs('results/model_round{}_{}'.format(constants.round_num, constants.strategy_type))
    for repeat_index in range(3,4): # constants.REPEAT_MAX
        for frame_index in range(len(constants.frame_list)):
            for model_index in range(4,len(constants.model_names)):
                frame = constants.frame_list[frame_index]
                start_time = time.time()
                train_model(constants, model_index, frame, repeat_index)
                elapsed_time = time.time() - start_time
                print('Elapsed Time:', elapsed_time/3600, 'hr')
                gc.collect()
