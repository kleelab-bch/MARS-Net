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
from data_processor import to3channel, preprocess_input, normalize_input, clip_input, heq_norm_input, aggregate_std_mean, get_std_mean_from_images, preprocess_per_input_image

class PredictDataGenerator:
    def __init__(self, img_path, mask_path, strategy_type, img_format='.png'):
        self.img_path = img_path
        self.mask_path = mask_path
        self.strategy_type = strategy_type
        self.img_format = img_format
        self.row, self.col = self.get_img_size()
        self.max_prev_frame_num = 4

    def preprocess_images(self, imgs, std_value, mean_value):
        imgs = imgs[:, np.newaxis, :, ...]

        if 'no_preprocessing' in str(self.strategy_type):
            imgs = to3channel(imgs)
        elif 'clip' in str(self.strategy_type):
            imgs = clip_input(imgs)
        elif 'normalize' in str(self.strategy_type):
            imgs = normalize_input(imgs)
        elif 'heq' in str(self.strategy_type):
            imgs = heq_norm_input(imgs)
        else:
            imgs = preprocess_input(imgs, std_value, mean_value)
            # imgs = preprocess_per_input_image(imgs)

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

        return imgs_list

    def get_expanded_whole_frames(self, std_value=None, mean_value=None):
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

        else:
            image_filenames = self.find_filenames(self.img_path)
            imgs, image_rows, image_cols = self.get_expanded_images(image_filenames)

        imgs_list = self.preprocess_images(imgs, std_value, mean_value)

        return imgs_list, image_filenames, image_cols, image_rows, self.col, self.row


    def get_expanded_images(self, namelist, ratio = 64.0, expand_more=True):
        # expand test set images because U-Net based models only takes the image of size in ratio of 64
        total_number = len(namelist)
        imgs_row_exp = int(np.ceil(np.divide(self.row, ratio) ) * ratio)
        imgs_col_exp = int(np.ceil(np.divide(self.col, ratio) ) * ratio)

        # For the images that need to be expanded more
        # necessary to prevent boundary effect during prediction
        if expand_more:
            if (imgs_row_exp - self.row) < ratio:
                imgs_row_exp = imgs_row_exp + int(ratio)

            if (imgs_col_exp - self.col) < ratio:
                imgs_col_exp = imgs_col_exp + int(ratio)

        if 'spheroid_test_VGG19' in str(self.strategy_type):  # created on 7/24/2021, to include padding on all sides
            imgs_row_exp = imgs_row_exp + int(ratio)
            imgs_col_exp = imgs_col_exp + int(ratio)

        print('imgs_exp row, col', imgs_row_exp, imgs_col_exp)

        imgs = np.ndarray((total_number, int(imgs_row_exp), int(imgs_col_exp)), dtype=np.uint8)

        i = 0
        for name in tqdm(namelist):
            # img = cv2.resize( cv2.imread(self.img_path + name, cv2.IMREAD_GRAYSCALE),(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC) # note that .resize takes (width, column) while .shape shows (height, width)
            img = cv2.imread(self.img_path + name, cv2.IMREAD_GRAYSCALE)
            if 'spheroid_test_VGG19' in str(self.strategy_type):  # created on 7/24/2021, to include padding on all sides
                imgs[i] = cv2.copyMakeBorder(img, int(ratio), imgs_row_exp - self.row - int(ratio), int(ratio), imgs_col_exp - self.col - int(ratio), cv2.BORDER_REFLECT)
            else:
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

        imgs = self.preprocess_images(imgs)

        return imgs, img_list, self.col, self.row

    def get_cropped_pathces_frames(self):
        print('get_cropped_pathces_frames')
        image_filenames = self.find_filenames(self.img_path)
        image_filenames = image_filenames[:1]
        # expand the original image, for instance:
        # 1644 --> 1664
        # 3286 --> 3328
        # 2048 --> 2048
        # 4096 --> 4096
        whole_frames, imgs_row_exp, imgs_col_exp = self.get_expanded_images(image_filenames, expand_more=False)
        whole_frames = self.preprocess_images(whole_frames)
        print('whole_frames', whole_frames.shape)
        whole_frame_rows = whole_frames.shape[2]
        whole_frame_cols = whole_frames.shape[3]

        # crop evenly the whole frame into NxN patches, referenced crop/crop_generator.py
        patch_size = 1024
        expand_size = 32
        pathces_per_frame = int(whole_frames.shape[0] * whole_frame_rows/patch_size * whole_frame_cols/patch_size)
        cropped_patches = np.zeros((pathces_per_frame, 3, patch_size, patch_size), dtype=np.float32)

        num_rows = int(whole_frame_rows / patch_size)
        num_cols = int(whole_frame_cols / patch_size)
        print('cropped_patches', cropped_patches.shape, num_rows, num_cols)

        patch_index = 0
        for a_frame in range(whole_frames.shape[0]):
            for row in range(num_rows):
                for col in range(num_cols):
                    # print(a_frame, row, col, patch_index)
                    cropped_patches[patch_index, :] = whole_frames[a_frame, :,
                                                                  row * patch_size:(row+1) * patch_size,
                                                                  col * patch_size:(col+1) * patch_size]
                    patch_index = patch_index + 1


        # These information is necessary to stitch the predicted patches and crop the stitched result
        self.stitch_num_rows = num_rows
        self.stitch_num_cols = num_cols
        self.stitch_whole_frames_shape = whole_frames.shape

        # expand each cropped patches
        expanded_cropped_patches = np.lib.pad(cropped_patches, ((0, 0), (0, 0), (expand_size, expand_size), (expand_size, expand_size)), 'symmetric')

        print('expanded_cropped_patches',expanded_cropped_patches.shape)
        assert expanded_cropped_patches.shape[2] == patch_size + 2*expand_size
        assert expanded_cropped_patches.shape[3] == patch_size + 2*expand_size

        return expanded_cropped_patches, image_filenames, expanded_cropped_patches.shape[3], expanded_cropped_patches.shape[2], patch_size, patch_size

    def stitch_cropped_patches_into_whole_frame(self, segmented_output):
        # Correct Test case: When I tested by inputting the cropped original imagge, the stitched output image was the same as the input
        print('stitch_cropped_patches_into_whole_frame', self.stitch_num_rows, self.stitch_num_cols)
        print('segmented_output', segmented_output.shape)
        patch_rows = segmented_output.shape[2]
        patch_cols = segmented_output.shape[3]

        # stitch patches to a whole frame
        whole_frames = np.zeros((self.stitch_whole_frames_shape[0], 1, self.stitch_whole_frames_shape[2], self.stitch_whole_frames_shape[3]), dtype=np.uint8)

        patch_index = 0
        for frame_counter in range(whole_frames.shape[0]):
            for row in range(self.stitch_num_rows):
                for col in range(self.stitch_num_cols):
                    whole_frames[frame_counter, :, row*patch_rows:(row+1)*patch_rows, col*patch_cols:(col+1)*patch_cols] = segmented_output[patch_index,:]
                    patch_index = patch_index + 1

        # crop the expanded whole frame based on get_cropped_pathces_frames()
        whole_frames = whole_frames[:,:,:self.row,:self.col]
        print('whole_frames', whole_frames.shape)

        return whole_frames

    # -------------------------------------------------------------------------
    # --------------------- Helper functions ----------------------------------
    # -------------------------------------------------------------------------

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