'''
Author Junbong Jang
Modified Date 2/2/2021

Facilitates prediction step of the segmentation pipeline
Gather test set and process it
'''
import numpy as np
import glob
import os, cv2
from data_processor import to3channel, preprocess_input, normalize_input, normalize_clip_input, heq_norm_input, aggregate_std_mean, get_std_mean_from_images


class DataGenerator:
    def __init__(self, img_path, n_frames_train, input_size, output_size, strategy_type, img_format = '.png'):
        self.n_frames_train = n_frames_train
        self.img_path = img_path
        self.input_size = input_size
        self.output_size = output_size
        self.strategy_type = strategy_type
        self.img_format = img_format
        self.row, self.col = self.get_img_size()

    def get_expanded_whole_frames(self):
        img_list = self.find_namespace()
        imgs, image_rows, image_cols = self.get_expanded_images(self.img_path, img_list)

        # ------------------- pre-processing images -------------------
        # std and mean from test set images
        std_value, mean_value = get_std_mean_from_images(self.img_path, img_format=self.img_format)
        print(mean_value, std_value)

        # std and mean from training set images, Don't use it because it yields worse prediction results
        # crop_path, _ = constants.get_crop_path(model_name, dataset_name, str(frame), str(0), str(repeat_index))
        # std_value, mean_value = aggregate_std_mean(constants.dataset_names, dataset_name, frame, repeat_index, crop_path)

        imgs = imgs[:, np.newaxis, :, :]
        if 'no_preprocessing' in str(self.strategy_type):
            imgs = to3channel(imgs)
        elif 'normalize_clip' in str(self.strategy_type):
            imgs = normalize_clip_input(imgs)
        elif 'normalize' in str(self.strategy_type):
            imgs = normalize_input(imgs)
        elif 'heq' in str(self.strategy_type):
            imgs = heq_norm_input(imgs)
        else:
            imgs = preprocess_input(imgs, std_value, mean_value)
        
        return imgs, img_list, image_cols, image_rows, self.col, self.row

    def get_orig_whole_frames(self):
        # used for Seg-Grad-CAM visualization
        img_list = self.find_namespace()
        total_number = len(img_list)
        imgs = np.ndarray((total_number, self.row, self.col), dtype=np.uint8)

        for i, name in enumerate(img_list):
            imgs[i] = cv2.imread(self.img_path + name, cv2.IMREAD_GRAYSCALE)

        # ------------------- pre-processing images -------------------
        std_value, mean_value = get_std_mean_from_images(self.img_path, img_format=self.img_format)

        print(mean_value, std_value)
        imgs = imgs[:, np.newaxis, :, :]

        if 'no_preprocessing' in str(self.strategy_type):
            imgs = imgs
        elif 'normalize_clip' in str(self.strategy_type):
            imgs = normalize_clip_input(imgs, mean_value, std_value)
        elif 'normalize' in str(self.strategy_type):
            imgs = normalize_input(imgs)
        else:
            imgs = preprocess_input(imgs, mean_value, std_value)

        return imgs, img_list, self.col, self.row

    def get_mask_frames(self, mask_path):
        img_namelist = self.find_namespace()
        masks = np.ndarray((len(img_namelist), self.row, self.col), dtype=np.uint8)
        for i, img_name in enumerate(img_namelist):
            image_id = img_name[-7:-4]

            mask_path_list = glob.glob(mask_path + '*' + self.img_format)
            mask_path_name = mask_path_list[0]
            mask_path_name = mask_path_name[:-7] + image_id + mask_path_name[-4:]

            masks[i] = cv2.imread(mask_path_name, cv2.IMREAD_GRAYSCALE)
        masks = masks/255

        return masks

    def find_namespace(self):
        img_list = []
        img_path = self.img_path
        
        img_filename_list = os.listdir(img_path)
        for img_filename in img_filename_list:
            if os.path.isfile(img_path + img_filename) and img_filename.endswith(self.img_format):
                img_list.append(img_filename)
        return img_list

    def get_img_size(self):
        img_path = self.img_path
        img_list = self.find_namespace()
        for file in img_list:
            if os.path.isfile(img_path + file) and file.endswith(self.img_format):
                return cv2.imread(img_path + file , cv2.IMREAD_GRAYSCALE).shape
        print("ERROR: get_img_size")
        return -1, -1

    def get_expanded_images(self,img_path, namelist, ratio = 64.0):
        # expand test set images because our model only takes the image of size in ratio of 64
        total_number = len(namelist)
        imgs_row_exp = int(np.ceil(np.divide(self.row, ratio) ) * ratio)
        imgs_col_exp = int(np.ceil(np.divide(self.col, ratio) ) * ratio)

        # crop images that are not expanded enough
        # this is necessary to prevent boundary effect
        if (imgs_row_exp - self.row) < ratio:
            imgs_row_exp = imgs_row_exp + int(ratio)
            print('imgs_row_exp', imgs_row_exp)

        if (imgs_col_exp - self.col) < ratio:
            imgs_col_exp = imgs_col_exp + int(ratio)
            print('imgs_col_exp', imgs_col_exp)

        imgs = np.ndarray((total_number, int(imgs_row_exp), int(imgs_col_exp)), dtype=np.uint8) 
        i = 0
        for name in namelist:
            img = cv2.resize( cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE),(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC)
            imgs[i] = cv2.copyMakeBorder(img, 0, imgs_row_exp - self.row, 0, imgs_col_exp - self.col, cv2.BORDER_REFLECT)
            i += 1
        return imgs, imgs_row_exp, imgs_col_exp
