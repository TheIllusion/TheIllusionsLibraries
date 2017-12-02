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

        self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)

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
        out_channels = in_channels
        self.batch_norm = nn.BatchNorm2d(out_channels)

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
                                                 kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, x):
        x = self.transpoed_conv(x)
        return x

# Dense Block - Figure 2.
class DenseBlock(nn.Module):
    def __init__(self, layers, in_channels, k_feature_maps, is_gpu_mode):
        super(DenseBlock, self).__init__()

        self.num_layers = layers
        self.layers_list = []

        # add layers to the list
        for i in xrange(layers):
            self.layers_list.append(Layer(kernel_size=3,
                                          in_channels=in_channels + i*k_feature_maps,
                                          out_channels=k_feature_maps))

        # gpu
        if is_gpu_mode:
            for model in self.layers_list:
                model.cuda()

    def forward(self, x):

        # feedforward x to the first layer and add the result to the list
        x_first_out = self.layers_list[0].forward(x)

        # initialize the list
        forwarded_output_list = []

        forwarded_output_list.append(x_first_out)
        prev_x = x

        # feedforward process from the second to the last layer
        for i in range(1, self.num_layers):
            # concatenate filters
            concatenated_filters = torch.cat((forwarded_output_list[i-1], prev_x), 1)
            # forward
            x_next_out = self.layers_list[i].forward(concatenated_filters)
            # add to the list
            forwarded_output_list.append(x_next_out)
            # prepare the temporary variable for the next loop
            prev_x = concatenated_filters

        # prepare the output (this will have (k_feature_maps * layers) feature maps)
        output_x = torch.cat(forwarded_output_list, 1)

        return output_x