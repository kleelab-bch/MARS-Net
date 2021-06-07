'''
Author Junbong Jang
Modified Date 2/2/2021

Facilitates prediction step of the segmentation pipeline
Gather test set and process it
'''
import numpy as np
import glob
import os, cv2
from tqdm import tqdm
from data_processor import to3channel, preprocess_input, normalize_input, normalize_clip_input, heq_norm_input, aggregate_std_mean, get_std_mean_from_images, preprocess_per_input_image

class PredictDataGenerator:
    def __init__(self, img_path, mask_path, strategy_type, img_format='.png'):
        self.img_path = img_path
        self.mask_path = mask_path
        self.strategy_type = strategy_type
        self.img_format = img_format
        self.row, self.col = self.get_img_size()
        self.max_prev_frame_num = 4

    def get_expanded_whole_frames(self):
        if 'temporal_' in str(self.strategy_type):
            image_filenames = self.find_filenames(self.mask_path)

            x_combined_filenames = []
            filename_remove_index = None
            for i, a_filename in enumerate(image_filenames):
                sampled_prev_img_filenames = self.regex_find_prev_filenames(a_filename, self.max_prev_frame_num)
                if sampled_prev_img_filenames is not None:
                    x_combined_filenames.append(a_filename)
                    for sampled_prev_img_filename in sampled_prev_img_filenames:
                        x_combined_filenames.append(sampled_prev_img_filename)
                else:
                    filename_remove_index = i

            if filename_remove_index is not None:
                del image_filenames[filename_remove_index]
            imgs, image_rows, image_cols = self.get_expanded_images(x_combined_filenames)

            new_imgs = np.zeros(shape=(int(imgs.shape[0]/(self.max_prev_frame_num+1)), (self.max_prev_frame_num+1), imgs.shape[1], imgs.shape[2]))
            for start_frame_num in range(self.max_prev_frame_num + 1):
                new_imgs[:, start_frame_num, :, :] = imgs[start_frame_num::(self.max_prev_frame_num + 1)]

            new_imgs = new_imgs[:, np.newaxis, :, :, :]
            imgs = new_imgs

        else:
            image_filenames = self.find_filenames(self.img_path)
            imgs, image_rows, image_cols = self.get_expanded_images(image_filenames)
            imgs = imgs[:, np.newaxis, :, :]
        # ------------------- pre-processing images -------------------
        # std and mean from test set images
        # std_value, mean_value = get_std_mean_from_images(self.img_path, img_format=self.img_format)
        # print(mean_value, std_value)

        # std and mean from training set images, Don't use it because it yields worse prediction results
        # crop_path, _ = constants.get_crop_path(model_name, dataset_name, str(frame), str(0), str(repeat_index))
        # std_value, mean_value = aggregate_std_mean(constants.dataset_names, dataset_name, frame, repeat_index, crop_path)

        if 'no_preprocessing' in str(self.strategy_type):
            imgs = to3channel(imgs)
        elif 'normalize_clip' in str(self.strategy_type):
            imgs = normalize_clip_input(imgs)
        elif 'normalize' in str(self.strategy_type):
            imgs = normalize_input(imgs)
        elif 'heq' in str(self.strategy_type):
            imgs = heq_norm_input(imgs)
        else:
            imgs = preprocess_input(imgs)

        print('imgs shape:', imgs.shape)
        if 'temporal_' in str(self.strategy_type):
            imgs_list = []
            for frame_counter in range(self.max_prev_frame_num + 1):
                imgs_list.append(imgs[:, :, frame_counter, :, :])

            # test by setting the previous frames same as the current frame
            # imgs = imgs[:, :, 0, :, :]
            # zero_imgs = np.ones_like(imgs)
            # imgs_list = [imgs, imgs, imgs, imgs, imgs]
        else:
            imgs_list = imgs

        return imgs_list, image_filenames, image_cols, image_rows, self.col, self.row


    def get_expanded_images(self, namelist, ratio = 64.0):
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
        for name in tqdm(namelist):
            # img = cv2.resize( cv2.imread(self.img_path + name, cv2.IMREAD_GRAYSCALE),(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC) # note that .resize takes (width, column) while .shape shows (height, width)
            img = cv2.imread(self.img_path + name, cv2.IMREAD_GRAYSCALE)
            imgs[i] = cv2.copyMakeBorder(img, 0, imgs_row_exp - self.row, 0, imgs_col_exp - self.col, cv2.BORDER_REFLECT)
            i += 1
        return imgs, imgs_row_exp, imgs_col_exp


    def get_orig_whole_frames(self):
        # used for Seg-Grad-CAM visualization
        img_list = self.find_filenames(self.img_path)
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

    def get_mask_frames(self):
        img_namelist = self.find_filenames(self.img_path)
        masks = np.ndarray((len(img_namelist), self.row, self.col), dtype=np.uint8)
        for i, img_name in enumerate(img_namelist):
            image_id = img_name[-7:-4]
            mask_path_list = glob.glob(self.mask_path + '*' + self.img_format)
            mask_path_name = mask_path_list[0]
            mask_path_name = mask_path_name[:-7] + image_id + mask_path_name[-4:]

            masks[i] = cv2.imread(mask_path_name, cv2.IMREAD_GRAYSCALE)
        masks = masks/255

        return masks

    def find_filenames(self, file_path):
        filtered_filename_list = []

        filename_list = os.listdir(file_path)
        for a_filename in filename_list:
            if os.path.isfile(file_path + a_filename) and a_filename.endswith(self.img_format):
                filtered_filename_list.append(a_filename)
        filtered_filename_list.sort()
        
        return filtered_filename_list

    def get_img_size(self):
        img_path = self.img_path
        img_list = self.find_filenames(img_path)
        for file in img_list:
            if os.path.isfile(img_path + file) and file.endswith(self.img_format):
                print('get_img_size', cv2.imread(img_path + file , cv2.IMREAD_GRAYSCALE).shape)
                return cv2.imread(img_path + file , cv2.IMREAD_GRAYSCALE).shape
        print("ERROR: get_img_size")
        return -1, -1

    def regex_find_prev_filenames(self, cur_filename, max_prev_frame_num):
        # For the given current frame, get n previous frames
        cur_frame_id = int(cur_filename[-7:-4])
        frame_interval = 1

        prev_filenames = []
        for prev_counter in range(frame_interval, frame_interval*(max_prev_frame_num+1), frame_interval):
            if cur_frame_id - frame_interval * max_prev_frame_num < 1:
                # TODO instead of appending cur_filename, append the last previous frame
                prev_filenames.append(cur_filename)
            else:
                prev_filename = f"{cur_filename[:-7]}{(cur_frame_id - prev_counter):03d}{cur_filename[-4:]}"
                prev_filenames.append(prev_filename)

        return prev_filenames