# simple vanilla autoencoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
#import data_loader
import numpy as np
import time

# gpu mode
is_gpu_mode = True

# batch size
BATCH_SIZE = 1

INPUT_IMAGE_WIDTH = 32
INPUT_IMAGE_HEIGHT = 32

# learning rate
LEARNING_RATE = 1 * 1e-6
  
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
                                                 kernel_size=4, stride=2, padding=1, output_padding=0, bias=True)

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
        
        self.first_conv_layer = TransitionDown(in_channels=3, out_channels=3, kernel_size=3)
        self.first_t_up_layer = TransitionUp(in_channels=3, out_channels=3)
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        
        x = self.first_conv_layer(x)
        x = self.first_t_up_layer(x)
        
        output = 255 * nn.functional.sigmoid(x)

        return output

if __name__ == "__main__":
    print 'main'

    autoencoder_model = AutoEncoder()

    if is_gpu_mode:
        autoencoder_model.cuda()

    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=LEARNING_RATE)

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

    i = 0
    
    while True:

        if is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
        else:
            inputs = Variable(torch.from_numpy(input_img).float())

        outputs = autoencoder_model(inputs)

        # l1-loss between real and fake
        l1_loss = F.l1_loss(outputs, inputs)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        l1_loss.backward(retain_graph=False)

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        time.sleep(5)
        
        '''
        if i % 50 == 0:
            print '-----------------------------------------------'
            print 'iterations = ', str(i)
            print 'loss(l1)     = ', str(l1_loss)
                
        i += 1
        '''