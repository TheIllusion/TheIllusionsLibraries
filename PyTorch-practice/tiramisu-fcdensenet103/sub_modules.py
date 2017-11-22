import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Layer module from the paper - Table 1.
class Layer(nn.Module):
    def __init__(self, in_channels):

        super(Layer, self).__init__()

        self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=3, stride=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)

        return x

# Transition Down (TD) module from the paper - Table 1.
class TransitionDown(nn.Module):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()

        self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=1, stride=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = F.max_pool2d(input=x, kernel_size=2)
        x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)

        return x

# Transition Up (TU) module from the paper - Table 1.
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

        self.transpoed_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                                 kernel_size=3, stride=2, bias=True)

    def forward(self, x):
        x = self.transpoed_conv(x)
        return x

# Dense Block of 4 layers. - Figure 2.
class DenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(DenseBlock, self).__init__()

        first_layer = Layer(in_channels)

        second_layer = Layer(in_channels)

        third_layer = Layer(in_channels)

        fourth_layer = Layer(in_channels)

    def forward(self, x):

        # forward
        x_second_out = self.first_layer.forward(x)

        # concatenate the output and the previous input
        x_second_out_concat = torch.cat((x, x_second_out), 0)

        # forward
        x_third_out = self.second_layer.forward(x_second_out_concat)

        # concatenate the output and the previous input
        x_third_out_concat = torch.cat((x_second_out_concat, x_third_out), 0)

        # forward
        x_fourth_out = self.third_layer.forward(x_third_out_concat)

        # concatenate the output and the previous input
        x_fourth_out_concat = torch.cat((x_third_out_concat, x_fourth_out), 0)

        # forward
        x_fifth_out = self.fourth_layer.forward(x_fourth_out_concat)

        # concatenate the output and the previous input
        x_fifth_out_concat = torch.cat((x_fourth_out_concat, x_fifth_out), 0)

        return x_fifth_out_concat