import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# macbook pro
root_dir = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/fashion-mnist/data/'

'''
CIFA-10 dataset (Docs: https://www.cs.toronto.edu/~kriz/cifar.html)
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
'''
train_loader = DataLoader(datasets.CIFAR10(root_dir, train=True, transform=None, target_transform=None, download=True))
test_loader = DataLoader(datasets.CIFAR10(root_dir, train=False, transform=None, target_transform=None, download=True))

class CNNBasic(nn.Module):
    def __init__(self):

        # input size is 32x32x3 for cifar-10 dataset
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=2, bias=True)

        # size of the feature maps would be 4x4 at this point

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

if __name__ == "__main__":

    # create a cnn model
    cnn_basic_mode = CNNBasic()

    # load data from cifar10
    for data, target in train_loader:
        data, target = Variable(data, volatile=True), Variable(target)

    print 'hi'
