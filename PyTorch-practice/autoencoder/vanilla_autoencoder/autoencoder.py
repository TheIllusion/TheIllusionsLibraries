# vanilla autoencoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from logger import Logger
import data_loader
import numpy as np
import time

# gpu mode
is_gpu_mode = False

# batch size
BATCH_SIZE = 1
TOTAL_ITERATION = 1000000

# learning rate
LEARNING_RATE = 3 * 1e-4

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 10000

# Macbook 12'
MODEL_SAVING_DIRECTORY = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/autoencoder/vanilla_autoencoder/models/'
RESULT_IMAGE_DIRECTORY = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/autoencoder/vanilla_autoencoder/result_imgs/'
TENSORBOARD_DIRECTORY = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/autoencoder/vanilla_autoencoder/tfboard/'

# tensor-board logger
if not os.path.exists(MODEL_SAVING_DIRECTORY):
    os.mkdir(MODEL_SAVING_DIRECTORY)

if not os.path.exists(RESULT_IMAGE_DIRECTORY):
    os.mkdir(RESULT_IMAGE_DIRECTORY)

if not os.path.exists(TENSORBOARD_DIRECTORY):
    os.mkdir(TENSORBOARD_DIRECTORY)

logger = Logger(TENSORBOARD_DIRECTORY)

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
        self.first_conv_layer = TransitionDown(in_channels=3, out_channels=32, kernel_size=3)
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

if __name__ == "__main__":
    print 'main'

    autoencoder_model = AutoEncoder()

    if is_gpu_mode:
        autoencoder_model.cuda()

    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=LEARNING_RATE)

    # read imgs
    image_buff_read_index = 0

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))

    # opencv style
    output_img_opencv = np.empty(shape=(data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT, 3))

    for i in range(TOTAL_ITERATION):

        exit_notification = False

        for j in range(BATCH_SIZE):
            data_loader.is_main_alive = True

            while data_loader.buff_status[image_buff_read_index] == 'empty':
                if exit_notification == True:
                    break

                time.sleep(1)
                if data_loader.buff_status[image_buff_read_index] == 'filled':
                    break

            if exit_notification == True:
                break

            np.copyto(input_img[j], data_loader.input_buff[image_buff_read_index])

            data_loader.buff_status[image_buff_read_index] = 'empty'

            image_buff_read_index = image_buff_read_index + 1
            if image_buff_read_index >= data_loader.image_buffer_size:
                image_buff_read_index = 0

        if is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
        else:
            inputs = Variable(torch.from_numpy(input_img).float())

        outputs = autoencoder_model(inputs)

        # l1-loss between real and fake
        l1_loss = F.l1_loss(inputs, outputs)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        l1_loss.backward(retain_graph=False)

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        if i % 50 == 0:
            print '-----------------------------------------------'
            print '-----------------------------------------------'
            print 'iterations = ', str(i)
            print 'loss(l1)     = ', str(l1_loss)
            print '-----------------------------------------------'

            # tf-board (scalar)
            #logger.scalar_summary('loss-l1', l1_loss, i)

            # tf-board (images - first 1 batches)

            output_imgs_temp = outputs.cpu().data.numpy()[0:1]
            input_imgs_temp = input_img[0:1]

            input_imgs_temp = input_imgs_temp[..., [2,1,0]]
            output_imgs_temp = output_imgs_temp[..., [2,1,0]]

            logger.image_summary('input', input_imgs_temp, i)
            logger.image_summary('generated', output_imgs_temp, i)