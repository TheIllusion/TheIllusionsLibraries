from __future__ import print_function, division
import os
import torch
#import pandas as pd
from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os, glob
import cv2

INPUT_IMAGE_WIDTH = 65
INPUT_IMAGE_HEIGHT = 65

#sample_memoization_dict = {}

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
        
        #global sample_memoization_dict
        
        #self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.jpg_files = glob.glob(os.path.join(root_dir,'*.jpg'))
        self.last_ok_img = None
        #self.sample_memoization_dict = sample_memoization_dict

    def __len__(self):
        return len(self.jpg_files)

    def __getitem__(self, idx):
        try:
            image = io.imread(self.jpg_files[idx])
            sample = {'image': image}
            if self.transform:
                sample = self.transform(sample)
            self.last_ok_sample = sample
        except:
            print("error file: ", self.jpg_files[idx], "idx: ", idx)
            sample = self.last_ok_img
        
        return sample

# Transforms

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        #assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        #image, landmarks = sample['image'], sample['landmarks']
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        #landmarks = landmarks * [new_w / w, new_h / h]

        #return {'image': img, 'landmarks': landmarks}
        return {'image': img}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        '''
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        '''
        
        self.output_size = output_size

    def __call__(self, sample):
        '''
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}
        '''
        
        image = sample['image']
        h, w = image.shape[:2]
        new_h = self.output_size
        new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        
        return {'image': image}

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        '''
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
        '''
        
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}