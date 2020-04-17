from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
# from dataset.aug import *
import random 
import numpy as np 
import os 
import torch
import scipy.io as sio
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# rotate 90,180,270
def rotation(image):
    channel = image.shape[0]
    n = random.randint(1,3)
    length = image.shape[1]
    for _ in range(n):
        for c in range(channel):
            for i in range(length):
                for j in range(i+1,length):
                    temp = image[c][i][j]
                    image[c][i][j] = image[c][j][i]
                    image[c][j][i] = temp
            for i in range(len(image[c])):
                image[c][i] = image[c][i][::-1]
    return image

# define dataset
class Hyperspectral_Dataset(Dataset):
    def __init__(self,config,train=True):
        self.config = config
        self.patch_size = self.config.patch_size
        self.train = train
        self.dataset = self.config.dataset
        self.path = os.path.join(os.getcwd(),'Data',self.dataset)
        
        if self.config.inference == False:
            if self.config.maxTrain == False:
                self.train_image_list = open(self.path+'/'+str(self.config.train_percent)+'/train.txt').read().splitlines()
                self.test_image_list = open(self.path+'/'+str(self.config.train_percent)+'/test.txt').read().splitlines()
            else:
                self.train_image_list = open(self.path+'/'+str(self.config.max_trainData)+'/train.txt').read().splitlines()
                self.test_image_list = open(self.path+'/'+str(self.config.max_trainData)+'/test.txt').read().splitlines()
        self.mat_path = self.path + '/' + self.dataset
        
        self.input_mat = sio.loadmat(self.mat_path+'.mat')['data']
        if self.config.inference == False:
            self.target_mat = sio.loadmat(self.mat_path+'.mat')['gt']
        else:
            self.inference_image_list = open(self.path+'/Inference/inference.txt').read().splitlines()

        self.height = self.input_mat.shape[0]
        self.width = self.input_mat.shape[1]
        if self.config.model_name == 'CNN_2D':
            self.band = 1
        else:
            self.band = self.config.band

        # Scale the input between [0,1]
        self.input_mat = self.input_mat.astype(float)
        self.input_mat -= np.min(self.input_mat)
        self.input_mat /= np.max(self.input_mat)
        if self.config.model_name == 'CNN_3D':
            self.input_mat -= 0.5

        # Calculate the mean of each channel for normalization
        self.mean_array = np.ndarray(shape=(self.band,),dtype=np.float32)
        for i in range(self.band):
            self.mean_array[i] = np.mean(self.input_mat[:,:,i])
        
        self.transpose_array = np.transpose(self.input_mat,(2,0,1))
        
        if self.config.inference == False:
            print('\tNumber of train data:{}, Number of test data:{}'.format(len(self.train_image_list), len(self.test_image_list)))
        else:
            print('\tNumber of inference data:{}'.format(len(self.inference_image_list)))

    def __getitem__(self,index):
        if self.config.inference == False:
            if self.train == True:
                patch_center = self.train_image_list[index].split(' ')
                data_name = patch_center[0]
                h = int(patch_center[1])
                w = int(patch_center[2])
                patch = self.Patch_Center(h,w)
                label = self.target_mat[h,w]-1
                # # Data augmentation
                # num = random.randint(0,1)
                # if num == 0 :
                #     patch = patch[:,::-1,:] # Flip patch up-down
                # if num == 1 :
                #     patch = patch[:,:,::-1] # Flip patch left-right
                # if num == 2 :
                #     self.patch[index] = rotation(self.patch[index]) # Rotate patch for 90/180/270
            else:
                patch_center = self.test_image_list[index].split(' ')
                h = int(patch_center[1])
                w = int(patch_center[2])
                patch = self.Patch_Center(h,w)
                label = self.target_mat[h,w]-1
        else:
            patch_center = self.inference_image_list[index].split(' ')
            h = int(patch_center[1])
            w = int(patch_center[2])
            patch = self.Patch_Center(h,w)
            return patch, h, w
        return patch, label.astype(np.int64)

    def __len__(self):
        if self.config.inference == False:
            if self.train == True:
                return len(self.train_image_list)
            else:
                return len(self.test_image_list)
        else:
            return len(self.inference_image_list)

    def Patch_Center(self,height_index,width_index):
        """
        Returns a mean-normalized patch, the center corner of which 
        is at (height_index, width_index)
        
        Inputs: 
        height_index - row index of the top left corner of the image patch
        width_index - column index of the top left corner of the image patch
        
        Outputs:
        mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE) 
        whose top left corner is at (height_index, width_index)
        """
        patch = np.zeros((self.band, self.patch_size, self.patch_size))
        offset = (self.patch_size-1)//2
        h_index = 0; w_index = 0
        for h in range(height_index-offset, height_index+offset+1):
            for w in range(width_index-offset, width_index+offset+1):
                if h<0 or h>=self.height or w<0 or w>=self.width:
                    continue
                else:
                    patch[:,h-height_index+offset,w-width_index+offset] = self.transpose_array[:,h,w]
        mean_normalized_patch = []
        for i in range(patch.shape[0]):
            mean_normalized_patch.append(patch[i] - self.mean_array[i]) 
        return np.array(mean_normalized_patch)
    
    def Patch_Center_1D(self,height_index,width_index):
        # when input dataset is crop_43_1D or crop_59_1D, then only use the spectral information to classify pixels.
        """
        Returns a mean-normalized patch, the center corner of which 
        is at (height_index, width_index)
        
        Inputs: 
        height_index - row index of the top left corner of the image patch
        width_index - column index of the top left corner of the image patch
        
        Outputs:
        mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE) 
        whose top left corner is at (height_index, width_index)
        """
        patch = np.zeros((self.band, 1, 1))
        offset = (1-1)//2
        h_index = 0; w_index = 0
        for h in range(height_index-offset, height_index+offset+1):
            for w in range(width_index-offset, width_index+offset+1):
                if h<0 or h>=self.height or w<0 or w>=self.width:
                    continue
                else:
                    patch[:,h-height_index+offset,w-width_index+offset] = self.transpose_array[:,h,w]
        mean_normalized_patch = []
        for i in range(patch.shape[0]):
            mean_normalized_patch.append(patch[i] - self.mean_array[i]) 
        return np.array(mean_normalized_patch)

    
if __name__ == '__main__':
    # train_dataloader = DataLoader(Hyperspectral_Dataset(config,train=True), \
    #                                 batch_size=config.batch_size,shuffle=True)
    # for iter,(train_images,train_labels) in enumerate(train_dataloader):
    #     print(iter, train_images.shape, train_labels.shape, len(train_labels))
    
    # test_dataloader = DataLoader(Hyperspectral_Dataset(config,train=False), \
    #                                 batch_size=config.batch_size,shuffle=False)
    # for iter,(test_images,test_labels) in enumerate(test_dataloader):
    #     print(iter, test_images.shape, test_labels.shape)

    inference_dataloader = DataLoader(Hyperspectral_Dataset(config,train=True), \
                                    batch_size=config.batch_size,shuffle=False)
    for iter,(inference_images) in enumerate(inference_dataloader):
        print(iter, inference_images.shape)