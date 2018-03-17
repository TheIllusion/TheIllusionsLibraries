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
import itertools

# gpu mode
is_gpu_mode = True

# batch size
BATCH_SIZE = 50
TOTAL_ITERATION = 1000000

# learning rate
LEARNING_RATE_GEN = 3 * 1e-4
LEARNING_RATE_DISC = 1 * 1e-4

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 10000

TARGET_SYSTEM_LIST = ['MACBOOK_12', 't005']
TARGET_SYSTEM = TARGET_SYSTEM_LIST[1]

if TARGET_SYSTEM == 'MACBOOK_12':
    # Macbook 12'
    MODEL_SAVING_DIRECTORY = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/GANs/lsgan_with_local_and_global_disc/models/'
    RESULT_IMAGE_DIRECTORY = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/GANs/lsgan_with_local_and_global_disc/result_imgs/'
    TENSORBOARD_DIRECTORY = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/GANs/lsgan_with_local_and_global_disc/tfboard/'
elif TARGET_SYSTEM == 't005':
    # t005
    MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/lsgan_with_local_and_global_disc/models/'
    RESULT_IMAGE_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/lsgan_with_local_and_global_disc/result_imgs/'
    TENSORBOARD_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/lsgan_with_local_and_global_disc/tfboard/'
else:
    exit(0)

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

# Transition Up (TU) module from the paper of Tiramisu - Table 1.
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, kernel_size):
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
                                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                                 output_padding=0, bias=False)

        # weight initialization
        torch.nn.init.xavier_uniform(self.transpoed_conv.weight)

    def forward(self, x):
        x = self.transpoed_conv(x)
        return x

# the last layer is fully connected
class GlobalDiscriminator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """

        super(GlobalDiscriminator, self).__init__()

        # input image will have the size of 64x64x3
        self.first_conv_layer = TransitionDown(in_channels=3, out_channels=32, kernel_size=5)
        self.second_conv_layer = TransitionDown(in_channels=32, out_channels=32, kernel_size=5)
        self.third_conv_layer = TransitionDown(in_channels=32, out_channels=64, kernel_size=5)
        self.fourth_conv_layer = TransitionDown(in_channels=64, out_channels=64, kernel_size=5)

        self.fc1 = nn.Linear(5 * 5 * 64, 1)

        torch.nn.init.xavier_uniform(self.fc1.weight)

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

        #print 'x.shape=', x.shape
        x = x.view(-1, 5 * 5 * 64)
        x = F.relu(self.fc1(x))

        sigmoid_out = nn.functional.sigmoid(x)

        return sigmoid_out

# the last layer is convolutional
class LocalDiscriminator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """

        super(LocalDiscriminator, self).__init__()

        # input image will have the size of 128x128x3
        self.first_conv_layer = TransitionDown(in_channels=3, out_channels=32, kernel_size=5)
        self.second_conv_layer = TransitionDown(in_channels=32, out_channels=64, kernel_size=5)
        self.third_conv_layer = TransitionDown(in_channels=64, out_channels=128, kernel_size=5)
        self.fourth_conv_layer = TransitionDown(in_channels=128, out_channels=32, kernel_size=5)
        self.fifth_conv_layer = TransitionDown(in_channels=32, out_channels=1, kernel_size=5)
        
        '''
        self.fc1 = nn.Linear(4 * 4 * 512, 10)
        self.fc2 = nn.Linear(10, 1)

        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        '''

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

        '''
        x = x.view(-1, 4 * 4 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        '''

        sigmoid_out = nn.functional.sigmoid(x)

        return sigmoid_out

# generator model (input random vector z will have the size of 4x4x3)
class Generator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Generator, self).__init__()

        # output feature map will have the size of 8x8
        self.first_deconv = TransitionUp(in_channels=3, out_channels=256, stride=2, padding=1, kernel_size=4)
        self.first_batch_norm = nn.BatchNorm2d(256)

        # output feature map will have the size of 16x16
        self.second_deconv = TransitionUp(in_channels=256, out_channels=128, stride=2, padding=1, kernel_size=4)
        self.second_batch_norm = nn.BatchNorm2d(128)

        # output feature map will have the size of 32x32
        self.third_deconv = TransitionUp(in_channels=128, out_channels=32, stride=2, padding=1, kernel_size=4)
        self.third_batch_norm = nn.BatchNorm2d(32)

        # output feature map will have the size of 64x64
        self.fourth_deconv = TransitionUp(in_channels=32, out_channels=16, stride=2, padding=1, kernel_size=4)
        self.fourth_batch_norm = nn.BatchNorm2d(16)

        # output feature map will have the size of 128x128
        self.fifth_deconv = TransitionUp(in_channels=16, out_channels=16, stride=1, padding=1, kernel_size=4)
        self.fifth_batch_norm = nn.BatchNorm2d(16)

        # output feature map will have the size of 128x128
        self.sixth_deconv = TransitionUp(in_channels=16, out_channels=16, stride=1, padding=1, kernel_size=4)
        self.sixth_batch_norm = nn.BatchNorm2d(16)

        # output feature map will have the size of 128x128
        self.seventh_deconv = TransitionUp(in_channels=16, out_channels=3, stride=1, padding=1, kernel_size=4)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x = self.first_deconv(x)
        x = self.first_batch_norm(x)
        x = F.leaky_relu(x)

        x = self.second_deconv(x)
        x = self.second_batch_norm(x)
        x = F.leaky_relu(x)

        x = self.third_deconv(x)
        x = self.third_batch_norm(x)
        x = F.leaky_relu(x)

        x = self.fourth_deconv(x)
        x = self.fourth_batch_norm(x)

        x = self.fifth_deconv(x)
        x = self.fifth_batch_norm(x)

        x = self.sixth_deconv(x)
        x = self.sixth_batch_norm(x)

        x = self.seventh_deconv(x)

        # sigmoid_out = nn.functional.sigmoid(x)
        tanh_out = nn.functional.tanh(x)

        out = (tanh_out + 1) * 255 / 2

        # print 'out.shape =', out.shape

        return out


if __name__ == "__main__":
    print 'main'

    global_disc = GlobalDiscriminator()
    local_disc = LocalDiscriminator()
    generator = Generator()

    if is_gpu_mode:
        global_disc.cuda()
        local_disc.cuda()
        generator.cuda()

    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE_GEN)

    disc_params_total = itertools.chain(local_disc.parameters(), global_disc.parameters())
    optimizer_disc = torch.optim.Adam(disc_params_total, lr=LEARNING_RATE_DISC)

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

        # random noise z
        noise_z = torch.randn(BATCH_SIZE, 3, 4, 4)
        #print 'noise_z=', noise_z
        
        if is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
            noise_z = Variable(noise_z.cuda())
        else:
            inputs = Variable(torch.from_numpy(input_img).float())
            noise_z = Variable(noise_z)

        # feedforward the inputs. generator
        outputs_gen = generator(noise_z)

        # feedforward the inputs. discriminator
        output_global_disc_real = global_disc(inputs)
        output_global_disc_fake = global_disc(outputs_gen)

        output_local_disc_real = local_disc(inputs)
        output_local_disc_fake = local_disc(outputs_gen)

        # lsgan loss for the discriminator
        loss_global_disc = 0.4 * (torch.mean((output_global_disc_real - 1) ** 2) + torch.mean(output_global_disc_fake ** 2))
        loss_local_disc = 0.6 * (torch.mean((torch.mean(output_local_disc_real) - 1) ** 2) + torch.mean(torch.mean(output_local_disc_fake) ** 2))

        loss_disc_total = loss_global_disc + loss_local_disc

        # lsgan loss for the generator
        loss_gen = 0.5 * torch.mean((output_global_disc_fake - 1) ** 2) + \
                   0.5 * torch.mean((torch.mean(output_local_disc_fake) - 1) ** 2)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer_disc.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_disc_total.backward(retain_graph=True)

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer_disc.step()

        # generator
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if i % 20 == 0:
            print '-----------------------------------------------'
            print 'iterations = ', str(i)
            print 'output_global_disc_real=', output_global_disc_real[0]
            print 'output_global_disc_fake=', output_global_disc_fake[0]
            print 'output_local_disc_real=', output_local_disc_real[0]
            print 'output_local_disc_fake=', output_local_disc_fake[0]
            print 'loss_global_disc=', loss_global_disc
            print 'loss_local_disc=', loss_local_disc
            print 'torch.mean((output_local_disc_fake - 1) ** 2)=', torch.mean((output_local_disc_fake - 1) ** 2)
            print 'loss_gen=', loss_gen

            # tf-board (scalar)
            #logger.scalar_summary('output_global_disc_real=', output_global_disc_real, i)
            #logger.scalar_summary('output_global_disc_fake=', output_global_disc_fake, i)
            #logger.scalar_summary('loss_global_disc=', loss_global_disc, i)
            #logger.scalar_summary('loss_local_disc=', loss_local_disc, i)
            #logger.scalar_summary('loss_gen=', loss_gen, i)

            # tf-board (images - first 2 batches)
            output_imgs_temp = outputs_gen.cpu().data.numpy()[0:2]
            input_imgs_temp = input_img[0:2]

            # rgb to bgr
            input_imgs_temp = input_imgs_temp[:, [2, 1, 0], ...]
            output_imgs_temp = output_imgs_temp[:, [2, 1, 0], ...]

            logger.image_summary('input', input_imgs_temp, i)
            logger.image_summary('generated', output_imgs_temp, i)
            logger.image_summary('input(duplicated)', input_imgs_temp, i)