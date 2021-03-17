import numpy as np
import os, cv2
import glob

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')
#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

import sys
sys.path.append('..')
from UserParams import UserParams
constants = UserParams('crop')


def preprocess_output(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    imgs_p = imgs_p.astype('float32')

    imgs_p /= 255.  # scale masks to [0, 1]
    return imgs_p


def preprocess_input(imgs, img_rows, img_cols, mean, std):
    print('preprocess_input')
    imgs_p = expand_channel_input(imgs, img_rows, img_cols)

    imgs_p -= mean
    imgs_p /= std
    return imgs_p


def normalize_input(imgs, img_rows, img_cols):
    print('normalize_input', imgs.shape)
    imgs_p = expand_channel_input(imgs, img_rows, img_cols)

    imgs_p /= 255.  # scale image to [0, 1]

    return imgs_p


def heq_norm_input(imgs, img_rows, img_cols):
    print('heq_norm_input', imgs.shape)
    imgs_heq = np.ndarray(imgs.shape, dtype=np.uint8)
    for img_index in range(imgs_heq.shape[0]):
        imgs_heq[img_index,0] = cv2.equalizeHist(imgs[img_index,0])

    imgs_p = expand_channel_input(imgs_heq, img_rows, img_cols)
    imgs_p /= 255.  # scale image to [0, 1]

    return imgs_p


def normalize_clip_input(imgs, img_rows, img_cols, mean, std):
    print('normalize_clip_input')
    imgs_p = expand_channel_input(imgs, img_rows, img_cols)

    max_val = mean + 3 * std
    min_val = mean - 3 * std
    if min_val < 0:
        min_val = 0
    if max_val > 255:
        max_val = 255
    print('min, max:', min_val, max_val)
    np.clip(imgs_p, min_val, max_val, out=imgs_p)
    # min max normalize
    imgs_p = (imgs_p - min_val) / (max_val - min_val)

    return imgs_p


def expand_channel_input(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], 3, img_rows, img_cols), dtype=np.uint8)
    print('expand_channel_input', imgs_p.shape)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 1] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 2] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    print('resized images shape: ', imgs_p.shape)
    imgs_p = imgs_p.astype('float32')

    return imgs_p


def augment_data(imgs,msks,edgs,batch_size,iteration):
    datagen = ImageDataGenerator(
        rotation_range=50.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')
      
    imgs = imgs[:,np.newaxis,:,:]
    msks = msks[:,np.newaxis,:,:]
    edgs = edgs[:,np.newaxis,:,:]
        
    train = np.zeros((iteration*batch_size, 1, imgs.shape[2], imgs.shape[3])).astype('uint8')
    mask = np.zeros((iteration*batch_size, 1, msks.shape[2], msks.shape[3])).astype('uint8')
    edge = np.zeros((iteration*batch_size, 1, edgs.shape[2], msks.shape[3])).astype('uint8')
    
    print('Data Generating...', batch_size, iteration)
    for samples in range(iteration): 
        for imags_batch in datagen.flow(imgs, batch_size=batch_size, seed = samples): #probably can change "activation" parameter
            break 
        for mask_batch in datagen.flow(msks, batch_size=batch_size, seed = samples): 
            break
        for edge_batch in datagen.flow(edgs, batch_size=batch_size, seed = samples): 
            break
        train[samples*batch_size:(samples+1)*batch_size] = imags_batch
        mask[samples*batch_size:(samples+1)*batch_size] = mask_batch
        edge[samples*batch_size:(samples+1)*batch_size] = edge_batch
    
    train = np.vstack([imgs, train])
    mask = np.vstack([msks, mask])
    edge = np.vstack([edgs, edge])
    
    mask = mask[:,:,30:imgs.shape[2]-30,30:imgs.shape[2]-30]
    edge = edge[:,:,30:imgs.shape[2]-30,30:imgs.shape[2]-30]
    
    train, mask, edge = shuffle(train, mask, edge, random_state=10)
    return train, mask, edge

class data_generate:
    def __init__(self, dataset_name, n_frames_train, input_size, output_size, random_seed, img_format, crop_mode, rand_crop_num, root, img_folder, mask_folder):
        self.n_frames_train = n_frames_train
        self.dataset_name = dataset_name
        self.random_seed = random_seed
        self.input_size = input_size
        self.output_size = output_size
        self.img_format = img_format
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.rand_crop_num = rand_crop_num
        self.crop_mode = crop_mode
        self.root = root
        
        self.row, self.col, self.total_frames = self.get_row_col()
        self.n_frames_test = self.total_frames - self.n_frames_train
    #==================================================================================
    # Get the size of image and number of images
    #==================================================================================
    def get_row_col(self):
        if constants.round_num == 1:
            m_path = self.root + self.dataset_name + self.mask_folder
        else:
            m_path = self.mask_folder
        mask_list = glob.glob(m_path + '*' + self.img_format)
        img = cv2.imread(mask_list[0], cv2.IMREAD_GRAYSCALE)
        r, c = img.shape
        if constants.round_num > 1:
            r=r-30
            c=c-30
        return float(r), float(c), len(mask_list)
    #==================================================================================
    # Get the size of image and number of images
    #==================================================================================
    def read_msk(self, msk_f):
        msk = cv2.imread(msk_f, cv2.IMREAD_GRAYSCALE)       
        msk[msk>0] = 255
        edg = cv2.Canny(msk,100,200)
        edg[edg>0] = 255
        return msk, edg
    #==================================================================================
    #==================================================================================
    def r_img_msk(self):
        if constants.round_num == 1:
            r_path = self.root + self.dataset_name + self.img_folder
            m_path = self.root + self.dataset_name + self.mask_folder
        else:
            r_path = self.root + self.dataset_name + self.img_folder
            m_path = self.mask_folder
        img_list = glob.glob(r_path + '*' + self.img_format)
        mask_list = glob.glob(m_path + '*' + self.img_format)

        total_number = len(mask_list)
        imgs = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        msks = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        edgs = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        framenames = list()

        for i in range(len(mask_list)):
            img_path = img_list[i]
            mask_path = mask_list[i]
            img_name = img_path[len(r_path):]
            mask_name = mask_path[len(r_path):]

            image_id = mask_name[-7:-4]
            img_name = img_name[:-7] + image_id + img_name[-4:]
            
            framenames.append(image_id)
            if constants.round_num == 1:
                msks[i], edgs[i] = self.read_msk(mask_list[i])
                imgs[i] = cv2.imread(r_path + img_name, cv2.IMREAD_GRAYSCALE)
            elif constants.round_num > 1:
                mask_orig, edge_orig = self.read_msk(mask_list[i])
                img_orig = cv2.imread(r_path + img_name, cv2.IMREAD_GRAYSCALE)
                row, col = img_orig.shape
                # because predicted images' border is hazy.
                imgs[i] = img_orig[30:, 30:]
                msks[i] = mask_orig[30:, 30:]
                edgs[i] = edge_orig[30:, 30:]

        return imgs,msks,edgs,framenames

    #==================================================================================
    #==================================================================================
    def split_val(self, inputs):
        test_size = self.total_frames - self.n_frames_train
        print('test size:', test_size, ' train size:', self.n_frames_train)
        if test_size > 0:
            train_0, test_0, train_1, test_1, train_2, test_2 = train_test_split(inputs[0], inputs[1], inputs[2],
                                                                                test_size = test_size, 
                                                                              random_state = self.random_seed)
        else:
            train_0, test_0, train_1, test_1, train_2, test_2 = inputs[0], [], inputs[1], [], inputs[2], []


        return train_0, test_0, train_1, test_1, train_2, test_2
    #==================================================================================
    #==================================================================================
    def sample_loc(self, edge, number, on_edge = True):
        kernel = np.ones((int(self.output_size/2), int(self.output_size/2)), np.uint8)
        dilate_Edge = cv2.dilate(edge, kernel, iterations=1)
        if on_edge:
            loc = np.where( dilate_Edge > 0 )
        else:
            loc = np.where( dilate_Edge < 1 )
        index = np.argmax([len(np.unique(loc[0])), len(np.unique(loc[1])) ])   
        sample_image_loc = np.random.choice(np.unique(loc[index]), number, replace = False)
        sample_pos = []
        for i in sample_image_loc:
            temp_index = np.where(loc[index] == i)[0]
            sample_pos.extend(np.random.choice(temp_index, 1))
        
        return loc, sample_pos
        
    def crop_on_loc(self, inputs, loc, sample):
        image, mask, edge = inputs[0], inputs[1], inputs[2]
        
        imgs = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8)
        msks = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgs = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8) 
    
        for i in range(len(sample)):
            imgs[i] = image[loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                            loc[1][sample[i]] :loc[1][sample[i]] + self.input_size]
            msks[i] =  mask[loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                            loc[1][sample[i]] :loc[1][sample[i]] + self.input_size]
            edgs[i] =  edge[loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                            loc[1][sample[i]] :loc[1][sample[i]] + self.input_size]
        return imgs, msks, edgs
        
    def crop_rand(self, inputs, edge_ratio = 0.6):
        image, mask, edge = inputs[0], inputs[1], inputs[2]
        
        edge_number = int(self.rand_crop_num*edge_ratio)
        back_number = self.rand_crop_num - edge_number
        
        loc_p, sample_p = self.sample_loc(edge, edge_number, on_edge = True)
        loc_n, sample_n = self.sample_loc(edge, back_number, on_edge = False)
        # pad and bias
        bound_in = int(np.ceil(self.input_size/2))
        
        
        image = np.lib.pad(image,((bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        mask = np.lib.pad(mask,((bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        edge = np.lib.pad(edge,((bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        
        imgs_p, msks_p, edgs_p = self.crop_on_loc([image, mask, edge], loc_p, sample_p)
        imgs_n, msks_n, edgs_n = self.crop_on_loc([image, mask, edge], loc_n, sample_n)
        return np.r_[imgs_p, imgs_n], np.r_[msks_p, msks_n], np.r_[edgs_p, edgs_n]
    #==================================================================================
    #==================================================================================
    def pad_img(self, inputs):
        num_y = int(np.ceil(self.col/self.output_size))
        num_x = int(np.ceil(self.row/self.output_size))
        sym = int(np.ceil(self.input_size/2 - self.output_size/2))
        # print('pad_img', num_y, num_x, sym)
        for i in range(3):
            inputs[i] = np.lib.pad(inputs[i], ((0,0),(0, int(num_x*self.output_size - inputs[i].shape[1])),(0, int(num_y*self.output_size - inputs[i].shape[2]))), 'symmetric')
            inputs[i] = np.lib.pad(inputs[i], ((0,0),(sym, sym), (sym, sym)),'symmetric')
        return inputs[0], inputs[1], inputs[2]

    def crop_even(self, image, mask, edge):
        image, mask, edge = self.pad_img([image, mask, edge])
        num_y = int(np.ceil(self.col / self.output_size))
        num_x = int(np.ceil(self.row / self.output_size))

        imgCrop = np.ndarray((self.n_frames_train * num_x * num_y, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        mskCrop = np.ndarray((self.n_frames_train * num_x * num_y, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgCrop = np.ndarray((self.n_frames_train * num_x * num_y, int(self.input_size), int(self.input_size)), dtype=np.uint8)


        for row in range(num_y):
            for col in range(num_x):
                for a_frame in range(self.n_frames_train):
                    imgCrop[col * num_y + row] = image[a_frame, col * self.output_size:col * self.output_size + self.input_size,
                                                 row * self.output_size:row * self.output_size + self.input_size]
                    mskCrop[col * num_y + row] = mask[a_frame, col * self.output_size:col * self.output_size + self.input_size,
                                                 row * self.output_size:row * self.output_size + self.input_size]
                    edgCrop[col * num_y + row] = edge[a_frame, col * self.output_size:col * self.output_size + self.input_size,
                                                 row * self.output_size:row * self.output_size + self.input_size]

        return imgCrop, mskCrop, edgCrop
    #==================================================================================
    #==================================================================================
    def crop_random(self, image, mask, edge):
        imgs_r = np.ndarray((self.n_frames_train*self.rand_crop_num, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        msks_r = np.ndarray((self.n_frames_train*self.rand_crop_num, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgs_r = np.ndarray((self.n_frames_train*self.rand_crop_num, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        
        for i in range(self.n_frames_train):
            imgs_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num], \
            msks_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num], \
            edgs_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num] = self.crop_rand([image[i], mask[i], edge[i]])

        return imgs_r, msks_r, edgs_r
    
    def crop(self):
        image, mask, edge, framenames = self.r_img_msk()

        imgsTrain_index, imgsTest_index, msksTrain_index, msksTest_index, edgsTrain_index, edgsTest_index = self.split_val([range(image.shape[0]), range(image.shape[0]), range(image.shape[0])])
        imgsTrain, imgsTest, msksTrain, msksTest, edgsTrain, edgsTest = self.split_val([image, mask, edge])
        if self.crop_mode == 'random':
            imgs_train,msks_train,edgs_train = self.crop_random(imgsTrain, msksTrain, edgsTrain)
        elif self.crop_mode == 'even':
            imgs_train,msks_train,edgs_train = self.crop_even(imgsTrain, msksTrain, edgsTrain)
        else:
            print('Crop Mode Error:', self.crop_mode)
            exit()
        avg = np.mean(imgsTrain)
        std = np.std(imgsTrain)
        
        return imgs_train, msks_train, edgs_train, avg, std, imgsTrain_index, imgsTest_index, framenames