'''
Author Junbong Jang
Date 3/14/2021

preprocess cropped images by shuffling, splitting (training, validation, and test sets) and standardizing
prepare generator to lazily load them

referenced: https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
'''
from data_processor import preprocess_input, preprocess_output, normalize_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')

from tqdm import tqdm
import glob
import re
import random
import numpy as np
import pickle
import cv2

from joblib import Parallel, delayed


def get_data_generators(round_num, dataset_names, model_name, frame, repeat_index, img_format, batch_size, process_type, save_path):
    # first set and retrieve image data in filenames
    set_training_dataset_names(round_num, dataset_names, model_name, frame, repeat_index, img_format, save_path)
    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = get_training_dataset_names(model_name, frame, repeat_index, save_path)

    # augment images in memory
    train_x, train_y = getAugmentedImages(x_train_filenames, y_train_filenames, batch_size, process_type)
    valid_x, valid_y = getAugmentedImages(x_val_filenames, y_val_filenames, batch_size, process_type)

    return train_x, train_y, valid_x, valid_y

    # augmentation generator
    # training_batch_generator = GetAugmentGenerator(x_train_filenames, y_train_filenames, batch_size, process_type, img_path, img_format, repeat_index)
    # validation_batch_generator = GetAugmentGenerator(x_val_filenames, y_val_filenames, batch_size, process_type, img_path, img_format, repeat_index)

    # It takes too long to train
    # training_batch_generator = CustomImageGenerator(x_train_filenames, y_train_filenames, batch_size, process_type, repeat_index, 3)
    # validation_batch_generator = CustomImageGenerator(x_val_filenames, y_val_filenames, batch_size, process_type, repeat_index, 3)

    # return training_batch_generator, validation_batch_generator


def get_cropped_filenames(round_num, dataset_names, frame, repeat_index, img_format):
    all_img_filenames = []
    all_mask_filenames = []

    for dataset_name in dataset_names:
        crop_path = '../crop/generated/crop_round{}_{}/'.format(round_num, dataset_name)
        crop_path_img = crop_path + f'img_repeat{repeat_index}/'
        crop_path_mask = crop_path + f'mask_repeat{repeat_index}/'

        img_filenames = glob.glob(crop_path_img + '*' + img_format)
        mask_filenames = glob.glob(crop_path_mask + '*' + img_format)
        assert len(img_filenames) == len(mask_filenames)

        # count unique frames
        unique_frames = []
        for img_filename in img_filenames:
            frame_id = regex_find_frame_id(img_filename)
            if frame_id not in unique_frames:
                unique_frames.append(frame_id)

        # randomly sample few frames
        sampled_frames = random.sample(unique_frames, k=frame)  # example output: ['f020', 'f010']

        # sampling img and mask filenames
        # filenames for image and mask are the same
        sampled_img_filenames = []
        for img_filename in img_filenames:
            for sampled_frame in sampled_frames:
                if sampled_frame in img_filename:
                    sampled_img_filenames.append(img_filename)
                    break
        all_img_filenames = all_img_filenames + sampled_img_filenames

        sampled_mask_filenames = []
        for mask_filename in mask_filenames:
            for sampled_frame in sampled_frames:
                if sampled_frame in mask_filename:
                    sampled_mask_filenames.append(mask_filename)
                    break
        all_mask_filenames = all_mask_filenames + sampled_mask_filenames

    return all_img_filenames, all_mask_filenames


def set_training_dataset_names(round_num, dataset_names, model_name, frame, repeat_index, img_format, save_path):
    x_filenames, y_filenames = get_cropped_filenames(round_num, dataset_names, frame, repeat_index, img_format)

    # split data
    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(
        x_filenames, y_filenames, shuffle=True, test_size=0.2, random_state=repeat_index)

    # shuffle them
    x_train_filenames, y_train_filenames = unison_shuffle_lists(x_train_filenames, y_train_filenames)
    x_val_filenames, y_val_filenames = unison_shuffle_lists(x_val_filenames, y_val_filenames)

    assert len(x_train_filenames) == len(y_train_filenames)
    assert len(x_val_filenames) == len(y_val_filenames)

    # Saving
    with open('{}x_train_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(x_train_filenames, fp)
    with open('{}x_val_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(x_val_filenames, fp)
    with open('{}y_train_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(y_train_filenames, fp)
    with open('{}y_val_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'wb') as fp:
        pickle.dump(y_val_filenames, fp)


def get_training_dataset_names(model_name, frame, repeat_index, save_path):
    with open('{}x_train_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        x_train_filenames = pickle.load(fp)
    with open('{}x_val_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        x_val_filenames = pickle.load(fp)
    with open('{}y_train_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        y_train_filenames = pickle.load(fp)
    with open('{}y_val_filenames_{}_frame{}_repeat{}.pickle'.format(save_path, model_name, frame, repeat_index), 'rb') as fp:
        y_val_filenames = pickle.load(fp)

    assert len(x_train_filenames) == len(y_train_filenames)
    assert len(x_val_filenames) == len(y_val_filenames)

    return x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames


def getAugmentedImages(x_filenames, y_filenames, batch_size, process_type):
    augmentation_factor = calc_augmentation_factor(x_filenames)
    batch_x, batch_y = GetImageMask(x_filenames, y_filenames)

    aug_images = np.zeros((augmentation_factor * batch_size, batch_x.shape[1], batch_x.shape[2], batch_x.shape[3])) # .astype('uint8')
    aug_masks = np.zeros((augmentation_factor * batch_size, batch_y.shape[1], batch_y.shape[2], batch_y.shape[3])) # .astype('uint8')
    print('getAugmentedImages orig', batch_x.shape, batch_x.dtype, batch_y.shape, batch_x.dtype)
    print('getAugmentedImages aug', aug_images.shape,aug_images.dtype, aug_masks.shape, aug_masks.dtype)

    # ------------ Augmentation -------------

    datagen = ImageDataGenerator(
        rotation_range=50.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        data_format='channels_first',
        fill_mode='reflect')

    for iteration in range(augmentation_factor):
        for aug_image_batch in datagen.flow(batch_x, batch_size=batch_size, seed=iteration):
            break
        for aug_mask_batch in datagen.flow(batch_y, batch_size=batch_size, seed=iteration):
            break
        aug_images[iteration * batch_size:(iteration + 1) * batch_size] = aug_image_batch
        aug_masks[iteration * batch_size:(iteration + 1) * batch_size] = aug_mask_batch

    aug_images = np.vstack([batch_x, aug_images])
    aug_masks = np.vstack([batch_y, aug_masks])

    # ---------- preprocessing ------------
    if process_type == 'normalize':
        aug_images = normalize_input(aug_images)
    elif process_type == 'standardize':
        aug_images = preprocess_input(aug_images)
    else:
        raise Exception('incorrect process_type {}'.format(process_type))

    aug_masks = preprocess_output(aug_masks)

    aug_images, aug_masks = unison_shuffle_ndarrays(aug_images, aug_masks)

    return aug_images, aug_masks


def GetImageMask(x_filenames, y_filenames):
    # first load all images in the filenames and create generator for augmenting them
    batch_x = np.asarray(read_images(x_filenames))
    batch_y = np.asarray(read_images(y_filenames))

    print('Input Data:', batch_x.shape, batch_y.shape)

    batch_x = batch_x[:,np.newaxis,:,:]
    batch_y = batch_y[:,np.newaxis,:,:]

    batch_y = batch_y[:, :, 30:batch_y.shape[2] - 30, 30:batch_y.shape[2] - 30]
    print('Preprocessed Data:', batch_x.shape, batch_y.shape)

    return batch_x, batch_y


def calc_augmentation_factor(x_filenames):
    '''
    4/7/2021
    Limit augmentation to utilize only about 25GB RAM memory
    Gradually reduce it as the number of frames increase
    :return:
    '''
    print('calc_augmentation_factor', len(x_filenames))
    return 400


def read_images(image_path_list):
    # https://stackoverflow.com/questions/33778155/python-parallelized-image-reading-and-preprocessing-using-multiprocessing
    images = Parallel(n_jobs=4, verbose=1)(
        delayed(cv2.imread)(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_path_list
    )
    return images


def unison_shuffle_lists(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a = [a[i] for i in p]
    b = [b[i] for i in p]
    return a, b


def unison_shuffle_ndarrays(a, b):
    assert len(a) == len(b)

    shuffler = np.random.permutation(len(a))
    a_shuffled = a[shuffler]
    b_shuffled = b[shuffler]

    return a_shuffled, b_shuffled


def regex_find_frame_id(filename):
    regex = re.compile(r'/f\d+_c')
    frame_id = regex.findall(filename)[0]  # example: '/f040_c'

    return frame_id

# ----------------------------------------------------------------------------------
# ------ old method that augments new images for every epoch on the fly ------------
# Cons: lower performance and much longer training time

def GetAugmentGenerator(x_filenames, y_filenames, batch_size, process_type, img_path, img_format, repeat_index):
    # first load all images in the filenames and create generator for augmenting them
    batch_x, batch_y = GetImageMask(x_filenames, y_filenames)

    # standardizing or normalizing data
    # if process_type == 'normalize':
    #     batch_x = normalize_input(batch_x)
    # elif process_type == 'standardize':
    #     std_value, mean_value = get_std_mean_from_images(img_path, img_format=img_format)
    #     batch_x = preprocess_input(batch_x, mean_value, mean_value)
    # else:
    #     raise Exception('incorrect process_type {}'.format(process_type))

    # batch_y = preprocess_output(batch_y)

    datagen = ImageDataGenerator(
        rotation_range=50.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        data_format='channels_first',
        fill_mode='reflect')

    img_generator = datagen.flow(batch_x, batch_size=batch_size, seed=repeat_index)
    mask_generator = datagen.flow(batch_y, batch_size=batch_size, seed=repeat_index)

    generator = MergedGenerators(img_generator, mask_generator)

    return generator


def combine_generator(gen1, gen2):
    while True:
        yield(next(gen1), next(gen2))


class MergedGenerators(Sequence):
    def __init__(self, *generators):
        self.generators = generators
        # TODO add a check to verify that all generators have the same length

    def __len__(self):
        return len(self.generators[0])

    def __getitem__(self, index):
        x_gen = self.generators[0]
        y_gen = self.generators[1]
        return x_gen[index], y_gen[index]


# ------ old method that uses generator to get few image patches and augment them on the fly ------------
class CustomImageGenerator(Sequence):
    # load data in batches for data that is too large to fit in RAM memory

    def __init__(self, x_filenames, y_filenames, batch_size, process_type, repeat_index, augmentation_factor):
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.batch_size = batch_size
        self.process_type = process_type
        self.repeat_index = repeat_index
        self.augmentation_factor = int(augmentation_factor)

        # get image size
        x_img = cv2.imread(x_filenames[0], cv2.IMREAD_GRAYSCALE)
        y_img = cv2.imread(y_filenames[0], cv2.IMREAD_GRAYSCALE)
        self.x_row, self.x_col = x_img.shape
        self.y_row, self.y_col = y_img.shape

        assert self.x_row == self.x_col
        assert self.y_row == self.y_col

    def __len__(self):
        # return (np.ceil(len(self.x_filenames) / float(self.batch_size))).astype(np.int)
        # longer list because of augmentation

        return (np.ceil(len(self.x_filenames) * self.augmentation_factor / float(self.batch_size))).astype(np.int)

    def __getitem__(self, batch_index):
        aug_batch_index = batch_index % self.augmentation_factor
        filenames_batch_index = int(batch_index / self.augmentation_factor)
        # print('__getitem__', batch_index, aug_batch_index, filenames_batch_index)
        if aug_batch_index == 0:
            batch_x_names = self.x_filenames[
                            filenames_batch_index * self.batch_size: (filenames_batch_index + 1) * self.batch_size]
            batch_y_names = self.y_filenames[
                            filenames_batch_index * self.batch_size: (filenames_batch_index + 1) * self.batch_size]

            batch_x = np.ndarray((self.batch_size, int(self.x_row), int(self.x_col)), dtype=np.uint8)
            batch_y = np.ndarray((self.batch_size, int(self.y_row), int(self.y_col)), dtype=np.uint8)

            for i, file_name in enumerate(batch_x_names):
                batch_x[i] = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            for j, file_name in enumerate(batch_y_names):
                batch_y[j] = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

            # augmentation_factor-1 is important for keeping original images into the list
            self.augmented_batch_x, self.augmented_batch_y = self.augment_data(batch_x, batch_y, self.repeat_index,
                                                                               (self.augmentation_factor-1))
            # self.save_augmented_images(self.augmented_batch_x, self.augmented_batch_y)

            # remove channel
            self.augmented_batch_x = self.augmented_batch_x[:,0,:,:]
            self.augmented_batch_y = self.augmented_batch_y[:,0,:,:]

            # standardizing or normalizing data
            if self.process_type == 'normalize':
                self.augmented_batch_x = normalize_input(self.augmented_batch_x)
            elif self.process_type == 'standardize':
                self.augmented_batch_x = preprocess_input(self.augmented_batch_x)
            else:
                raise Exception('incorrect process_type {}'.format(self.process_type))
            self.augmented_batch_y = preprocess_output(self.augmented_batch_y)

        return self.augmented_batch_x[aug_batch_index * self.batch_size: (aug_batch_index + 1) * self.batch_size], \
               self.augmented_batch_y[aug_batch_index * self.batch_size: (aug_batch_index + 1) * self.batch_size]

    def get_x_size(self):
        return self.x_row, self.x_col

    def get_y_size(self):
        return self.y_row, self.y_col

    def augment_data(self, orig_imgs, orig_masks, repeat_index, augmentation_factor):
        datagen = ImageDataGenerator(
            rotation_range=50.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            data_format='channels_first',
            fill_mode='reflect')

        orig_imgs = orig_imgs[:, np.newaxis, :, :]
        orig_masks = orig_masks[:, np.newaxis, :, :]
        # print('Augment Data ...', orig_imgs.shape, orig_masks.shape)

        all_augmented_images = np.zeros(( (augmentation_factor+1) * self.batch_size, 1, orig_imgs.shape[2], orig_imgs.shape[3])).astype('uint8')
        all_augmented_masks = np.zeros(( (augmentation_factor+1) * self.batch_size, 1, orig_masks.shape[2], orig_masks.shape[3])).astype('uint8')

        for iteration in range(augmentation_factor):
            for augmented_images in datagen.flow(orig_imgs, batch_size=self.batch_size, seed=iteration):
                break
            for augmented_masks in datagen.flow(orig_masks, batch_size=self.batch_size, seed=iteration):
                break

            all_augmented_images[iteration*self.batch_size:(iteration+1)*self.batch_size] = augmented_images
            all_augmented_masks[iteration*self.batch_size:(iteration+1)*self.batch_size] = augmented_masks

        all_augmented_images[(iteration+1) * self.batch_size:(iteration + 2) * self.batch_size] = orig_imgs
        all_augmented_masks[(iteration+1) * self.batch_size:(iteration + 2) * self.batch_size] = orig_masks

        all_augmented_masks = all_augmented_masks[:, :, 30:orig_imgs.shape[2] - 30, 30:orig_imgs.shape[2] - 30]

        return all_augmented_images, all_augmented_masks


    def save_augmented_images(self, batch_x, batch_y):
        # to debug image and mask mismatch 3/16/2021
        all_index_list = [x for x in range(batch_x.shape[0])]
        all_index_list = all_index_list[0:100] + all_index_list[-100:]

        for batch_index in all_index_list:
            x = batch_x[batch_index]
            x = np.moveaxis(x, 0, -1) # convert from first_channel to last_channel
            cv2.imwrite(f'results/history_round1_single_micro_VGG19_dropout/image_{batch_index}.png', x)
        for batch_index in all_index_list:
            y = batch_y[batch_index]
            y = np.moveaxis(y, 0, -1)
            cv2.imwrite(f'results/history_round1_single_micro_VGG19_dropout/mask_{batch_index}.png', y)

