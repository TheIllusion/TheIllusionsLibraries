# vanilla autoencoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TransitionDown, self).__init__()

        # self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, padding=1, stride=2, bias=True)

        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x = F.max_pool2d(input=x, kernel_size=2)
        # x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)

        return x

# Transition Up (TU) module
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
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

        self.transpoed_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        # weight initialization
        torch.nn.init.xavier_uniform(self.transpoed_conv.weight)

    def forward(self, x):
        x = self.transpoed_conv(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """

        super(AutoEncoder, self).__init__()

        # input image will have the size of 64x64x3
        self.first_conv_layer = TransitionDown(in_channels=6, out_channels=32, kernel_size=3)
        self.second_conv_layer = TransitionDown(in_channels=32, out_channels=64, kernel_size=3)
        self.third_conv_layer = TransitionDown(in_channels=64, out_channels=128, kernel_size=3)
        self.fourth_conv_layer = TransitionDown(in_channels=128, out_channels=256, kernel_size=3)
        self.fifth_conv_layer = TransitionDown(in_channels=256, out_channels=512, kernel_size=3)

        # transition ups
        self.sixth_t_up_layer = TransitionUp(in_channels=512, out_channels=256)
        self.seventh_t_up_layer = TransitionUp(in_channels=256, out_channels=128)
        self.eighth_t_up_layer = TransitionUp(in_channels=128, out_channels=64)
        self.ninth_t_up_layer = TransitionUp(in_channels=64, out_channels=32)
        self.tenth_t_up_layer = TransitionUp(in_channels=32, out_channels=3)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        x = self.first_conv_layer(x)
        x = self.second_conv_layer(x)
        x = self.third_conv_layer(x)
        x = self.fourth_conv_layer(x)
        x = self.fifth_conv_layer(x)
        x = self.sixth_t_up_layer(x)
        x = self.seventh_t_up_layer(x)
        x = self.eighth_t_up_layer(x)
        x = self.ninth_t_up_layer(x)
        x = self.tenth_t_up_layer(x)

        output = 255 * nn.functional.sigmoid(x)

        return output

