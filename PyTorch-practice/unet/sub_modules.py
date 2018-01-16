import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Layer module from the paper - Table 1.
class Layer(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(Layer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=1, bias=True)
        
        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)
        
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv(x))
        
        # try batch-norm (this could be different from the original u-net)
        x = self.batch_norm(x)
        
        return x

# max pooling
class TransitionDown(nn.Module):
    def __init__(self, in_channels):        
        super(TransitionDown, self).__init__()

    def forward(self, x): 
        x = F.max_pool2d(input=x, kernel_size=2)

        return x

# Transition Up (TU) module 
class TransitionUp(nn.Module):
    def __init__(self, in_channels):
        super(TransitionUp, self).__init__()

        '''
        # nn.ConvTranspose2d
        Args
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution
        padding (int or tuple, optional): Zero-padding added to both sides of the input
        output_padding (int or tuple, optional): Zero-padding added to one side of the output
        groups (int, optional): Number of blocked connections from input channels to output channels
        bias (bool, optional): If True, adds a learnable bias to the output
        dilation (int or tuple, optional): Spacing between kernel elements
        '''

        self.transpoed_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels/2,
                                                 kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)
        
        # weight initialization
        torch.nn.init.xavier_uniform(self.transpoed_conv.weight)

    def forward(self, x):
        x = self.transpoed_conv(x)
        return x
