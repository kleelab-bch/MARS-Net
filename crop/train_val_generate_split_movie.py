import numpy as np
import os, cv2
from os import path
import glob
#import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import backend as K
K.set_image_data_format('channels_first')
#K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
smooth = 1.

import sys
sys.path.append('..')
from UserParams import UserParams
constants = UserParams('crop')

def preprocess_output(imgs, img_rows, img_cols):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 1] = cv2.resize(imgs[i, 1], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 2] = cv2.resize(imgs[i, 2], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    imgs_p = imgs_p.astype('float32')
    imgs_p /= 255.  # scale masks to [0, 1]
    return imgs_p

def preprocess_input(imgs, img_rows, img_cols, mean, std):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 1] = cv2.resize(imgs[i, 1], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        imgs_p[i, 2] = cv2.resize(imgs[i, 2], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)

    imgs_p = imgs_p.astype('float32')
    imgs_p -= mean
    imgs_p /= std
    return imgs_p
    

def augment_data(imgs,msks,edgs,iteration):
    batch_size = 128  # how many images to augment at once
    datagen = ImageDataGenerator(
        rotation_range=50.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')
        
    train = np.zeros((iteration*batch_size, imgs.shape[1], imgs.shape[2], imgs.shape[3])).astype('uint8')
    mask = np.zeros((iteration*batch_size, msks.shape[1], msks.shape[2], msks.shape[3])).astype('uint8')
    edge = np.zeros((iteration*batch_size, edgs.shape[1], edgs.shape[2], msks.shape[3])).astype('uint8')
    
    print('Data Generating...')
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
    
    mask = mask[:,:,30:imgs.shape[2]-30,30:imgs.shape[3]-30]
    edge = edge[:,:,30:imgs.shape[2]-30,30:imgs.shape[3]-30]
    
    train, mask, edge = shuffle(train, mask, edge, random_state=10)
    return train, mask, edge


class data_generate:
    def __init__(self, dataset_folder, n_frames_train, input_size, output_size, random_seed, saved_folder, img_format = '.png', rand_crop_num = 200, root = '', img_folder = '/img/', mask_folder = '/mask/'):
        self.n_frames_train = n_frames_train
        self.dataset_folder = dataset_folder
        self.random_seed = random_seed
        self.input_size = input_size
        self.output_size = output_size
        self.img_format = img_format
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.rand_crop_num = rand_crop_num
        self.saved_folder = saved_folder
        self.root = root
        
        self.row, self.col, self.total_frames = self.get_row_col()
        self.n_frames_val = 0
        self.n_frames_test = self.total_frames - self.n_frames_train - self.n_frames_val
    #==================================================================================
    # Get the size of image and number of images
    #==================================================================================
    def get_row_col(self):
        path = self.root + self.dataset_folder + self.img_folder
        img_list = glob.glob(path + '*' + self.img_format)
        img = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
        r, c = img.shape
        if constants.round_num > 1:
            r=r-30
            c=c-30
        return float(r), float(c), len(img_list)
    #==================================================================================
    # Get the size of image and number of images
    #==================================================================================
    def read_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.repeat(img[np.newaxis, :, :], 3, axis=0)
        return img
        
        
    def read_msk(self, msk_f, expand_bool=True):
        msk = cv2.imread(msk_f, cv2.IMREAD_GRAYSCALE)       
        msk[msk>0] = 255
        edg = cv2.Canny(msk,100,200)
        edg[edg>0] = 255
        
        if expand_bool:
            msk = np.repeat(msk[np.newaxis, :, :], 3, axis=0)
            edg = np.repeat(edg[np.newaxis, :, :], 3, axis=0)
        
        return msk, edg
    #==================================================================================
    #==================================================================================
    def r_img_msk(self):
        r_path = self.root + self.dataset_folder + self.img_folder
        img_all_path = self.root + self.dataset_folder + '/img_all/'
        if constants.round_num == 1:
            m_path = self.root + self.dataset_folder + self.mask_folder
        else:
            m_path = self.mask_folder
        img_list = glob.glob(r_path + '*' + self.img_format)
        mask_list = glob.glob(m_path + '*' + self.img_format)

        total_number = len(mask_list)
        imgs = np.ndarray((total_number, 3, int(self.row), int(self.col)), dtype=np.uint8)
        msks = np.ndarray((total_number, 3, int(self.row), int(self.col)), dtype=np.uint8)
        edgs = np.ndarray((total_number, 3, int(self.row), int(self.col)), dtype=np.uint8)
        frame_list = []
        for i in range(len(mask_list)):
            print(i)
            img_path = img_list[i]
            mask_path = mask_list[i]
            
            an_img = self.read_img(img_path)
            an_mask, an_edge = self.read_msk(mask_path)
            
            # --- get previous and next images ------
            image_name = img_path[len(r_path):]
            frame_num_string = img_path[-7:-4]
            frame_number = int(frame_num_string)
            frame_list.append(frame_number)
            
            prev_img_name = image_name[:-7] + str(frame_number - 1).zfill(3) + image_name[-4:]
            next_img_name = image_name[:-7] + str(frame_number + 1).zfill(3) + image_name[-4:]
            
            print(prev_img_name)
            print(next_img_name)
            
            if frame_number == 1:
                an_img[2,:,:] = cv2.imread(img_all_path + next_img_name, cv2.IMREAD_GRAYSCALE)
                an_mask[2,:,:], an_edge[2,:,:] = self.read_msk(m_path + next_img_name, expand_bool=False)
                
            elif frame_number == 200:
                an_img[0,:,:] = cv2.imread(img_all_path + prev_img_name, cv2.IMREAD_GRAYSCALE)
                an_mask[0,:,:], an_edge[0,:,:] = self.read_msk(m_path + prev_img_name, expand_bool=False)
            
            else:
                an_img[0,:,:] = cv2.imread(img_all_path + prev_img_name, cv2.IMREAD_GRAYSCALE)
                an_img[2,:,:] = cv2.imread(img_all_path + next_img_name, cv2.IMREAD_GRAYSCALE)
                an_mask[0,:,:], an_edge[0,:,:] = self.read_msk(m_path + prev_img_name, expand_bool=False)
                an_mask[2,:,:], an_edge[2,:,:] = self.read_msk(m_path + next_img_name, expand_bool=False)
                
            # --- Crop Image if necessary -------
            if constants.round_num == 1:
                imgs[i] = an_img
                msks[i] = an_mask
                edgs[i] = an_edge
            if constants.round_num > 1:
                # because predicted images' border is hazy.
                imgs[i] = an_img[:, 30:, 30:]
                msks[i] = an_mask[:, 30:, 30:]
                edgs[i] = an_edge[:, 30:, 30:]
                
            
        np.save(self.saved_folder + self.dataset_folder  + '_' + str(self.n_frames_train) + '.npy', frame_list);
        return imgs,msks,edgs
    
    #==================================================================================
    #==================================================================================
    def split_val(self, inputs):
        t_n = self.total_frames
        test_size = t_n - self.n_frames_train
        print(self.n_frames_train)
        print(test_size)
        print()
        if test_size > 0:
            train_0, val_test_0, train_1, val_test_1, train_2, val_test_2 = train_test_split(inputs[0], inputs[1], inputs[2],
                                                                                test_size = test_size, 
                                                                              random_state = self.random_seed)
        else:
            train_0, val_test_0, train_1, val_test_1, train_2, val_test_2 = inputs[0], [], inputs[1], [], inputs[2], []
       
        #val_0, test_0, val_1, test_1, val_2, test_2 = train_test_split(val_test_0, val_test_1, val_test_2,
        #                                                                  test_size = t_n - self.n_frames_train - self.n_frames_val, 
        #                                                                  random_state = self.random_seed)
        val_0, test_0, val_1, test_1, val_2, test_2 = [], val_test_0, [], val_test_1, [], val_test_2
        return train_0, val_0, test_0, train_1, val_1, test_1, train_2, val_2, test_2
    #==================================================================================
    #==================================================================================
    def sample_loc(self, edge, number, on_edge = True):
        kernel = np.ones((int(self.output_size/2), int(self.output_size/2)), np.uint8)
        edge = edge[0,:,:]
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
        
        imgs = np.ndarray((len(sample), 3, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        msks = np.ndarray((len(sample), 3, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgs = np.ndarray((len(sample), 3, int(self.input_size), int(self.input_size)), dtype=np.uint8) 
        
        for i in range(len(sample)):
            imgs[i] = image[:, loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                               loc[1][sample[i]] :loc[1][sample[i]] + self.input_size]
            msks[i] =  mask[:, loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                               loc[1][sample[i]] :loc[1][sample[i]] + self.input_size]
            edgs[i] =  edge[:, loc[0][sample[i]] :loc[0][sample[i]] + self.input_size, 
                               loc[1][sample[i]] :loc[1][sample[i]] + self.input_size]
        return imgs, msks, edgs
        
    def crop_rand(self, inputs, edge_ratio = 0.6):
        image, mask, edge = inputs[0], inputs[1], inputs[2]
        
        edge_number = int(self.rand_crop_num*edge_ratio)
        back_number = self.rand_crop_num - edge_number
        
        loc_p, sample_p = self.sample_loc(edge, edge_number, on_edge = True)
        loc_n, sample_n = self.sample_loc(edge, back_number, on_edge = False)
        
        #pad and bias
        bound_in = int(np.ceil(self.input_size/2))
        image = np.lib.pad(image,((0, 0), (bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        mask = np.lib.pad(mask,((0, 0), (bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        edge = np.lib.pad(edge,((0, 0), (bound_in, bound_in), (bound_in, bound_in)),'symmetric')

        
        imgs_p, msks_p, edgs_p = self.crop_on_loc([image, mask, edge], loc_p, sample_p)
        imgs_n, msks_n, edgs_n = self.crop_on_loc([image, mask, edge], loc_n, sample_n)
        return np.r_[imgs_p, imgs_n], np.r_[msks_p, msks_n], np.r_[edgs_p, edgs_n]
    
    
    #==================================================================================
    #==================================================================================
    def crop_train(self, image, mask, edge):
        imgs_r = np.ndarray((self.n_frames_train*self.rand_crop_num, 3, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        msks_r = np.ndarray((self.n_frames_train*self.rand_crop_num, 3, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgs_r = np.ndarray((self.n_frames_train*self.rand_crop_num, 3, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        
        for i in range(self.n_frames_train):
            imgs_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num], msks_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num], edgs_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num] = self.crop_rand([image[i], mask[i], edge[i]])

        return imgs_r, msks_r, edgs_r
    
    def crop(self):
        image,mask,edge = self.r_img_msk()
        print(image.shape)
        print(mask.shape)
        print(edge.shape)
        imgsTrain_index, imgsVal_index, imgsTest_index, msksTrain_index, msksVal_index, msksTest_index, edgsTrain_index, edgsVal_index, edgsTest_index= self.split_val([range(image.shape[0]), range(image.shape[0]), range(image.shape[0])])
        
        imgsTrain, imgsVal, imgsTest, msksTrain, msksVal, msksTest, edgsTrain, edgsVal, edgsTest= self.split_val([image, mask, edge])
        imgs_train,msks_train,edgs_train = self.crop_train(imgsTrain, msksTrain, edgsTrain)

        avg = np.mean(imgsTrain)
        std = np.std(imgsTrain)
        
        return imgs_train, msks_train, edgs_train, avg, std, imgsTrain_index, imgsVal_index, imgsTest_index