import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time
import numpy as np
import data_loader

# gpu mode
is_gpu_mode = True

# batch size
BATCH_SIZE = 4
TOTAL_ITERATION = 1000000

# learning rate
LEARNING_RATE = 1 * 1e-4

# zero centered
MEAN_VALUE_FOR_ZERO_CENTERED = 128

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 10000

# i7-2600k
MODEL_SAVING_DIRECTORY = '/home/illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/GANs/models/'

# Transition Down (TD) module from the paper of Tiramisu - Table 1.
class TransitionDown(nn.Module):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()

        self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=1, stride=1, bias=True)

        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)

        out_channels = in_channels
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.max_pool2d(input=x, kernel_size=2)
        x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)

        return x

# Transition Up (TU) module from the paper of Tiramisu - Table 1.
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
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

# generator model (input random vector z will have the size of 4x4x3)
class Generator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Generator, self).__init__()

        # output feature map will have the size of 8x8x3
        self.first_deconv = TransitionUp(in_channels=3, out_channels=1024, kernel_size=2)

        # output feature map will have the size of 16x16x3
        self.second_deconv = TransitionUp(in_channels=1024, out_channels=512, kernel_size=4)

        # output feature map will have the size of 32x32x3
        self.third_deconv = TransitionUp(in_channels=512, out_channels=256, kernel_size=4)

        # output feature map will have the size of 64x64x3
        self.fourth_deconv = TransitionUp(in_channels=256, out_channels=3, kernel_size=4)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        # first try without relu
        x = self.first_deconv(x)
        x = self.second_deconv(x)
        x = self.third_deconv(x)
        x = self.fourth_deconv(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Discriminator, self).__init__()

        # input image will have the size of 64x64x3
        self.first_conv_layer = TransitionDown(kernel_size=3, in_channels=3, out_channels=128)
        self.second_conv_layer = TransitionDown(kernel_size=3, in_channels=3, out_channels=256)
        self.third_conv_layer = TransitionDown(kernel_size=3, in_channels=3, out_channels=512)

        self.fc1 = nn.Linear(512 * 4 * 4, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x = self.first_conv_layer(x)
        x = self.second_conv_layer(x)
        x = self.third_conv_layer(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

if __name__ == "__main__":
    print 'main'

    gen_model = Generator()
    disc_model = Discriminator()

    if is_gpu_mode:
        gen_model.cuda()
        disc_model.cuda()

    learning_rate = LEARNING_RATE

    optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=learning_rate)
    optimizer_disc = torch.optim.Adam(disc_model.parameters(), lr=learning_rate)

    # read imgs
    image_buff_read_index = 0

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    answer_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))

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
            input_imgs = Variable(torch.from_numpy(input_img).float().cuda())
        else:
            input_imgs = Variable(torch.from_numpy(input_img).float())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer_gen.zero_grad()
        optimizer_disc.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer_gen.step()
        optimizer_disc.step()

        if i % 10 == 0:
            print '-----------------------------------'
            print 'iterations = ', str(i)
            print 'loss = ', str(loss)

        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(gen_model.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'vanilla_gan_pytorch_iter_' + str(i) + '.pt')
