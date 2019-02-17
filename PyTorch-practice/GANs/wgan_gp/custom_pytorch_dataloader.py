from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os, glob
import cv2

INPUT_IMAGE_WIDTH = 65
INPUT_IMAGE_HEIGHT = 65

class FaceDataset(Dataset):
    """FaceDataset"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        self.jpg_files = glob.glob(os.path.join(root_dir,'*.jpg'))

    def __len__(self):
        return len(self.jpg_files)

    def __getitem__(self, idx):
        
        # Input Image
        input_img = cv2.imread(self.jpg_files[idx], cv2.IMREAD_COLOR)
        
        input_img_tmp = cv2.resize(input_img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)

        # PyTorch format
        input_buff = np.empty(shape=(3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

        input_buff[0, :, :] = input_img_tmp[:, :, 0]
        input_buff[1, :, :] = input_img_tmp[:, :, 1]
        input_buff[2, :, :] = input_img_tmp[:, :, 2]
        
        '''
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        '''
        sample = {'image': input_buff}

        if self.transform:
            sample = self.transform(sample)

        return sample
