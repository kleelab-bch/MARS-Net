import numpy as np
import os, cv2
import random
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

class prediction_data_generate:
    def __init__(self, img_path, msk_path, n_frames_train, input_size, output_size, 
    random_seed, img_format = '.png', rand_crop_num = 500):
        self.n_frames_train = n_frames_train
        self.img_path = img_path
        self.msk_path = msk_path
        self.random_seed = random_seed
        self.input_size = input_size
        self.output_size = output_size
        self.img_format = img_format
        self.rand_crop_num = rand_crop_num
        self.row, self.col = self.get_img_size()
        

    #==================================================================================
    #==================================================================================
    def get_testing_set(self):
        
        # get the namespace
        namespace = self.find_namespace()
        
        imgs = self.testing_read(self.img_path, namespace)
        #cropping
        imgs_val = self.crop_valid(imgs, len(namespace))
        image_cols, image_rows = self.get_img_size()
        return imgs_val, namespace, image_cols, image_rows 

    def get_whole_frames(self):
        
        # get the namespace
        namespace = self.find_namespace()
        imgs, masks, image_rows, image_cols = self.expand_read(self.img_path, self.msk_path, namespace)
        return imgs, masks, namespace, image_cols, image_rows 
    

    def crop_valid(self, image, n_frames_test): # crop the testing image for validation
        num_y = int(np.ceil((self.col+0.0)/self.output_size));
        num_x = int(np.ceil((self.row+0.0)/self.output_size));
        

        imgs = np.ndarray((n_frames_test*num_x*num_y, int(self.input_size), int(self.input_size)), dtype=np.uint8)
       
        for i in range(n_frames_test):
            imgs[i*num_x*num_y:(i+1)*num_x*num_y] = self.crop_even(image[i])
        
        imgs = imgs[:,np.newaxis,:,:]
        return imgs
    
    def crop_even(self, img): # previous design..
        image = self.pad_img(img)
        imgCrop = self.crp_e(image)
        return imgCrop
    
    def crp_e(self, img):
        num_y = int(np.ceil((self.col+0.0)/self.output_size));
        num_x = int(np.ceil((self.row+0.0)/self.output_size));
        bias = (self.input_size-self.output_size)/2
        
        imgCrop = np.ndarray((num_x*num_y, int(self.input_size), int(self.input_size)), dtype=np.uint8)

        for row in range(num_y):
            for col in range(num_x):
                imgCrop[col*num_y+row] = img[col*self.output_size:col*self.output_size+self.input_size, row*self.output_size:row*self.output_size+self.input_size]              
                
                
        return imgCrop
    
    def pad_img(self, image):
        num_y = int(np.ceil((self.col+0.0)/self.output_size));
        num_x = int(np.ceil((self.row+0.0)/self.output_size));
        sym = int(np.ceil(self.input_size/2 - self.output_size/2))

        
        image = np.lib.pad(image, ((0, int(num_x*self.output_size - image.shape[0])),(0, int(num_y*self.output_size - image.shape[1]))), 'symmetric')
        image = np.lib.pad(image, ((sym, sym), (sym, sym)),'symmetric');
        return image
    #==================================================================================
    #==================================================================================
    def get_training_set(self):
        # get the namespace
        namespace = self.find_namespace()
        # sample, and save the namelist  
        #namelist = self.name_sampling(self.n_frames_train, namespace)
        namelist = self.name_sampling_witerval(self.n_frames_train, namespace)
        # read the images 
        imgsTrain, msksTrain, edgsTrain = self.training_read(self.msk_path, self.img_path, namelist)
        # crop the images; in 3 ways ; evenly, randomly, edges 
        imgs_train,msks_train,edgs_train = self.crop_train(imgsTrain, msksTrain, edgsTrain)
        
        avg = np.mean(imgsTrain)
        std = np.std(imgsTrain)
        
        return imgs_train, msks_train, edgs_train, avg, std, namelist
    #==================================================================================
    #==================================================================================    
    def name_sampling_witerval(self, num, name_space): #Sampleing image with a certain interval
        total = len(name_space)
        name_space.sort()
        
        result_list = []
        interval = int(total/num)
        for i in range(0, total, interval):
            result_list.append(name_space[i])
        result_list.sort()
        return result_list


    #==================================================================================
    #==================================================================================
    def name_sampling(self, num, name_space): # reservoir sampling
        total = len(name_space)
        num = min(num, total)
        result_list = []
        for i in range(0, num):
            result_list.append(name_space[i])
        for i in range(num, total):
            ram = random.randint(0, i)
            if ram < num:
                result_list[ram] = name_space[i]
        result_list.sort()
        return result_list
    #==================================================================================
    #==================================================================================
    def find_namespace(self):
        namespace = []
        img_path = self.img_path
        
        img_file_name = os.listdir(img_path)
        for file in img_file_name:
            if os.path.isfile(img_path + file) and file.endswith(self.img_format):
                namespace.append(file)
        
        return namespace
    #==================================================================================
    #==================================================================================
    def check_namespace(self): # for training set, if all the files have raw image and labeling
        valid_list = []
        img_path = self.img_path
        msk_path = self.msk_path
        for file in self.namespace:
            if os.path.isfile(img_path + file) and os.path.isfile(msk_path + file) and file.endswith(self.img_format):
                valid_list.append(file)

        return valid_list
    #==================================================================================
    #==================================================================================
    def get_img_size(self): # for training set
        
        img_path = self.img_path
        msk_path = self.msk_path
        namespace = self.find_namespace()
        
        for file in namespace:
            #Find the mapping filename in the mask folder
            #filemask = 'mask' + file[-8:]
            filemask = file
            #filemaks = "%04d" % filemask + '.png'
            if os.path.isfile(img_path + file) and os.path.isfile(msk_path + filemask) and file.endswith(self.img_format):
                return cv2.imread(img_path + file , cv2.IMREAD_GRAYSCALE).shape
        print("invalid imgs")
        return -1, -1
    #==================================================================================
    #==================================================================================
    def crop_train(self, image, mask, edge): 
        num_y = int(np.ceil((self.col+0.0)/self.output_size));
        num_x = int(np.ceil((self.row+0.0)/self.output_size));
        
        imgs_r = np.ndarray((self.n_frames_train*self.rand_crop_num, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        msks_r = np.ndarray((self.n_frames_train*self.rand_crop_num, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgs_r = np.ndarray((self.n_frames_train*self.rand_crop_num, int(self.input_size), int(self.input_size)), dtype=np.uint8)
        
        for i in range(self.n_frames_train):
            imgs_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num], msks_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num], edgs_r[i*self.rand_crop_num:(i+1)*self.rand_crop_num] = self.crop_rand([image[i], mask[i], edge[i]])

        return imgs_r, msks_r, edgs_r
    #==================================================================================
    #==================================================================================
    def sample_loc(self, edge, number, on_edge = True):
        kernel = np.ones((int(self.output_size/2), int(self.output_size/2)), np.uint8)
        dilate_Edge = cv2.dilate(edge, kernel, iterations=1)
        if on_edge:
            loc = np.where( dilate_Edge > 0 )
        else:
            loc = np.where( dilate_Edge < 1 )
        sample_pos = np.random.randint(0, high = loc[0].shape[0], size = number)
        return loc, sample_pos
    #==================================================================================
    #==================================================================================
    def crop_on_loc(self, inputs, loc, sample):
        image, mask, edge = inputs[0], inputs[1], inputs[2]
        
        imgs = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8)
        msks = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8)
        edgs = np.ndarray((len(sample), int(self.input_size), int(self.input_size)), dtype=np.uint8) 
    
        for i in range(len(sample)):
            imgs[i] = image[loc[0][sample[i]]:loc[0][sample[i]]+self.input_size, 
                            loc[1][sample[i]]:loc[1][sample[i]]+self.input_size]
            msks[i] =  mask[loc[0][sample[i]]:loc[0][sample[i]]+self.input_size, 
                            loc[1][sample[i]]:loc[1][sample[i]]+self.input_size]
            edgs[i] =  edge[loc[0][sample[i]]:loc[0][sample[i]]+self.input_size, 
                            loc[1][sample[i]]:loc[1][sample[i]]+self.input_size]
        return imgs, msks, edgs
    #==================================================================================
    #==================================================================================
    def crop_rand(self, inputs, edge_ratio = 0.8):
        image, mask, edge = inputs[0], inputs[1], inputs[2]
        
        edge_number = int(self.rand_crop_num*edge_ratio)
        back_number = self.rand_crop_num - edge_number
        
        loc_p, sample_p = self.sample_loc(edge, edge_number, on_edge = True)
        loc_n, sample_n = self.sample_loc(edge, back_number, on_edge = False)
        #pad and bias
        bound_in = int(np.ceil(self.input_size/2.0))
        bias = (self.input_size-self.output_size)/2;
        
        image = np.lib.pad(image,((bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        mask = np.lib.pad(mask,((bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        edge = np.lib.pad(edge,((bound_in, bound_in), (bound_in, bound_in)),'symmetric')
        
        imgs_p, msks_p, edgs_p = self.crop_on_loc([image, mask, edge], loc_p, sample_p)
        imgs_n, msks_n, edgs_n = self.crop_on_loc([image, mask, edge], loc_n, sample_n)
        return np.r_[imgs_p, imgs_n], np.r_[msks_p, msks_n], np.r_[edgs_p, edgs_n]
    #==================================================================================
    #==================================================================================
    def testing_read(self,img_path, namelist, ratio = 64.0): # read images within namelist
        total_number = len(namelist)
        imgs_row_exp = int(np.ceil(np.divide(self.row, ratio) ) * ratio)
        imgs_col_exp = int(np.ceil(np.divide(self.col, ratio) ) * ratio)
        #applying space to save the results
        imgs = np.ndarray((total_number, int(imgs_row_exp), int(imgs_col_exp)), dtype=np.uint8) 
        i = 0
        for name in namelist:
            
            img = cv2.resize( cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE) ,(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC)
            imgs[i] = cv2.copyMakeBorder(img, 0, imgs_row_exp - self.row, 0, imgs_col_exp - self.col, cv2.BORDER_REFLECT)
            i += 1
        return imgs, imgs_row_exp, imgs_col_exp
    
    #==================================================================================
    #==================================================================================
    def expand_read(self,img_path, msk_path, namelist, ratio = 64.0): # read images within namelist
        total_number = len(namelist)
        imgs_row_exp = int(np.ceil(np.divide(self.row, ratio) ) * ratio)
        imgs_col_exp = int(np.ceil(np.divide(self.col, ratio) ) * ratio)
        #applying space to save the results
        imgs = np.ndarray((total_number, int(imgs_row_exp), int(imgs_col_exp)), dtype=np.uint8) 
        masks = np.ndarray((total_number, int(imgs_row_exp), int(imgs_col_exp)), dtype=np.uint8)
        i = 0
        for name in namelist:
            #Find the mapping filename in the mask folder
            #filemask = 'mask' + name[-8:]
            filemask = name
            # added 3/16/2020 by Junbong Jang
            img_orig = cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE)
            '''
            [row, col] = img_orig.shape
            img_orig = img_orig[30:row-30, 30:col-30]
            '''
            # ------------------
            img = cv2.resize( img_orig ,(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC)
            imgs[i] = cv2.copyMakeBorder(img, 0, imgs_row_exp - self.row, 0, imgs_col_exp - self.col, cv2.BORDER_REFLECT)
            
            mask = cv2.resize( cv2.imread(msk_path + filemask, cv2.IMREAD_GRAYSCALE) ,(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC)
            masks[i] = cv2.copyMakeBorder(mask, 0, imgs_row_exp - self.row, 0, imgs_col_exp - self.col, cv2.BORDER_REFLECT)
            
            i += 1
        return imgs, masks, imgs_row_exp, imgs_col_exp
    #==================================================================================
    #==================================================================================
    def training_read(self, msk_path, img_path, namelist): # read images within namelist
        total_number = len(namelist)
        
        #print(total_number)
        #print(self.row)
        #print(self.col)
        
        #applying space to save the results
        imgs = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8) 
        msks = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        edgs = np.ndarray((total_number, int(self.row), int(self.col)), dtype=np.uint8)
        
        i = 0
        for name in namelist:
            #Find the mapping filename in the mask folder
            #filemask = (int(name[12:14]) + 1 ) * 5
            #filemask = "%03d" % filemask + '.tif'
            msks[i], edgs[i] = self.read_msk(msk_path + name) 
            imgs[i] = cv2.resize( cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE) ,(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC)
            i += 1
        
        return imgs,msks,edgs
    #==================================================================================
    #==================================================================================
    def read_msk(self, msk_f, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))): # read the edge regions with the help of mask images
        msk = cv2.imread(msk_f, cv2.IMREAD_GRAYSCALE)       
        msk = cv2.morphologyEx(msk, cv2.MORPH_CLOSE, kernel)
        msk[msk>0] = 255
        msk = cv2.resize(msk,(int(self.col), int(self.row)), interpolation = cv2.INTER_CUBIC)  # why to resize? Xitong?
        msk[msk>0] = 255
        edg = cv2.Canny(msk,100,200)
        edg[edg>0] = 255

        return msk, edg
    #==================================================================================
    #==================================================================================
    