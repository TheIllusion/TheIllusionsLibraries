import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# gpu mode
is_gpu_mode = True 

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

class Unet(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Unet, self).__init__()

        # unet architecture
        
        # first conv
        self.first_conv_layer = Layer(kernel_size=3, in_channels=3, out_channels=64)
        
        # second conv
        self.second_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=64)
        
        # max pool
        self.first_transition_down = TransitionDown(64)
        
        # third conv
        self.third_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=128)
        
        # fourth conv
        self.fourth_conv_layer = Layer(kernel_size=3, in_channels=128, out_channels=128)
        
        # max pool
        self.second_transition_down = TransitionDown(128)
        
        # fifth conv
        self.fifth_conv_layer = Layer(kernel_size=3, in_channels=128, out_channels=256)
        
        # sixth conv
        self.sixth_conv_layer = Layer(kernel_size=3, in_channels=256, out_channels=256)
        
        # max pool
        self.third_transition_down = TransitionDown(256)
        
        # seventh conv
        self.seventh_conv_layer = Layer(kernel_size=3, in_channels=256, out_channels=512)
        
        # eighth conv
        self.eighth_conv_layer = Layer(kernel_size=3, in_channels=512, out_channels=512)
        
        # max pool
        self.fourth_transition_down = TransitionDown(512)
        
        # ninth conv
        self.ninth_conv_layer = Layer(kernel_size=3, in_channels=512, out_channels=1024)
        
        # tenth conv
        self.tenth_conv_layer = Layer(kernel_size=3, in_channels=1024, out_channels=1024)
        
        # first Transition Up
        self.later_first_transition_up = TransitionUp(1024)
        
        # followed by 2 conv layers
        self.later_first_conv_layer = Layer(kernel_size=3, in_channels=1024, out_channels=512)
        self.later_second_conv_layer = Layer(kernel_size=3, in_channels=512, out_channels=512)
        
        # second Transition Up
        self.later_second_transition_up = TransitionUp(512)
        
        # followed by 2 conv layers 
        self.later_third_conv_layer = Layer(kernel_size=3, in_channels=512, out_channels=256)
        self.later_fourth_conv_layer = Layer(kernel_size=3, in_channels=256, out_channels=256)
        
        # third Transition Up
        self.later_third_transition_up = TransitionUp(256)
        
        # followed by 2 conv layers 
        self.later_fifth_conv_layer = Layer(kernel_size=3, in_channels=256, out_channels=128)
        self.later_sixth_conv_layer = Layer(kernel_size=3, in_channels=128, out_channels=128)
        
        # fourth Transition Up
        self.later_fourth_transition_up = TransitionUp(128)
        
        # followed by 2 conv layers 
        self.later_seventh_conv_layer = Layer(kernel_size=3, in_channels=128, out_channels=64)
        self.later_eighth_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=64)
        
        # last conv layer (num. of output channels should be modified according to the number of classes)
        self.later_last_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=3)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        # unet
        ######################################################################
        # define the forward connections
        ######################################################################
        x = self.first_conv_layer(x)
        x_second_conv_out = self.second_conv_layer(x)
        
        # first TD
        x = self.first_transition_down(x_second_conv_out)
        
        x = self.third_conv_layer(x)
        x_fourth_conv_out = self.fourth_conv_layer(x)
        
        # second TD
        x = self.second_transition_down(x_fourth_conv_out)
        
        x = self.fifth_conv_layer(x)
        x_sixth_conv_out = self.sixth_conv_layer(x)
        
        # third TD
        x = self.third_transition_down(x_sixth_conv_out)
        
        x = self.seventh_conv_layer(x)
        x_eighth_conv_out = self.eighth_conv_layer(x)
        
        # fourth TD
        x = self.fourth_transition_down(x_eighth_conv_out)
        
        ######################################################################
        # define the middle connections
        ######################################################################
        x = self.ninth_conv_layer(x)
        x = self.tenth_conv_layer(x)
        
        ######################################################################
        # define the backward connections
        ######################################################################
        # first TU
        x = self.later_first_transition_up(x)
        x_later_first_concat = torch.cat((x, x_eighth_conv_out), 1)
        
        # followed by 2 conv layers
        x = self.later_first_conv_layer(x_later_first_concat)
        x = self.later_second_conv_layer(x)
        
        # second TU
        x = self.later_second_transition_up(x)
        x_later_second_concat = torch.cat((x, x_sixth_conv_out), 1)
        
        # followed by 2 conv layers
        x = self.later_third_conv_layer(x_later_second_concat)
        x = self.later_fourth_conv_layer(x)
        
        # third TU
        x = self.later_third_transition_up(x)
        x_later_third_concat = torch.cat((x, x_fourth_conv_out), 1)
        
        # followed by 2 conv layers
        x = self.later_fifth_conv_layer(x_later_third_concat)
        x = self.later_sixth_conv_layer(x)
        
        # fourth TU
        x = self.later_fourth_transition_up(x)
        x_later_fourth_concat = torch.cat((x, x_second_conv_out), 1)
        
        # followed by 2 conv layers
        x = self.later_seventh_conv_layer(x_later_fourth_concat)
        x = self.later_eighth_conv_layer(x)
        
        # last conv layer
        x = self.later_last_conv_layer(x)

        sigmoid_out = nn.functional.sigmoid(x)
        #softmax_out = custom_softmax_for_segmentation(x_last_conv_out, 3)

        output = sigmoid_out * 255
        #output = softmax_out

        return output
