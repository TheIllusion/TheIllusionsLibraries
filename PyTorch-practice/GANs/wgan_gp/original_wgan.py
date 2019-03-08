import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time, os
import numpy as np
import data_loader
from logger import Logger

# gpu mode
is_gpu_mode = True

# batch size
BATCH_SIZE = 60
TOTAL_ITERATION = 1000000

# learning rate
LEARNING_RATE_GENERATOR = 0.1 * 1e-4
LEARNING_RATE_DISCRIMINATOR = 0.1 * 1e-4

# zero centered
# MEAN_VALUE_FOR_ZERO_CENTERED = 128

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 10000

# t005
MODEL_SAVING_DIRECTORY = "//home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/wgan_gp/checkpoints/"

TF_BOARD_DIRECTOR = "/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/wgan_gp/tfboard/"

RESULT_IMAGE_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/wgan_gp/gen_images/'

if not os.path.exists(RESULT_IMAGE_DIRECTORY):
    os.mkdir(RESULT_IMAGE_DIRECTORY)

if not os.path.exists(MODEL_SAVING_DIRECTORY):
    os.mkdir(MODEL_SAVING_DIRECTORY)

if not os.path.exists(TF_BOARD_DIRECTOR):
    os.mkdir(TF_BOARD_DIRECTOR)
    
# tensor-board logger
logger = Logger(TF_BOARD_DIRECTOR)
    
class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TransitionDown, self).__init__()

        # self.drop_out = nn.Dropout2d(p=0.2)

        # GAN's doesn't work well with stride=2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, padding=1, stride=1, bias=False)

        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)

        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d((3, 3), stride=1)

    def forward(self, x):
        # x = F.max_pool2d(input=x, kernel_size=2)
        # x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)

        x = F.max_pool2d(input=x, kernel_size=2)

        return x

# Transition Up (TU) module from the paper of Tiramisu - Table 1.
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size):
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
                                                 kernel_size=kernel_size, stride=stride, padding=1, output_padding=0,
                                                 bias=False)

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
        self.first_deconv = TransitionUp(in_channels=3, out_channels=256, stride=2, kernel_size=4)
        #self.first_batch_norm = nn.BatchNorm2d(256)
        self.first_batch_norm = nn.InstanceNorm2d(256)
        self.decA1 = INSResBlock(256, 256)
        
        # output feature map will have the size of 16x16x3
        self.second_deconv = TransitionUp(in_channels=256, out_channels=128, stride=2, kernel_size=4)
        self.second_batch_norm = nn.InstanceNorm2d(128)
        self.decA2 = INSResBlock(128, 128)

        # output feature map will have the size of 32x32x3
        self.third_deconv = TransitionUp(in_channels=128, out_channels=64, stride=2, kernel_size=4)
        self.third_batch_norm = nn.InstanceNorm2d(64)
        self.decA3 = INSResBlock(64, 64)

        # output feature map will have the size of 64x64x3
        self.fourth_deconv = TransitionUp(in_channels=64, out_channels=32, stride=2, kernel_size=4)
        self.fourth_batch_norm = nn.InstanceNorm2d(32)

        # output feature map will have the size of 64x64x3
        self.fifth_deconv = TransitionUp(in_channels=32, out_channels=3, stride=1, kernel_size=4)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x = self.first_deconv(x)
        x = self.first_batch_norm(x)
        x = F.leaky_relu(x)
        x = self.decA1(x)

        x = self.second_deconv(x)
        x = self.second_batch_norm(x)
        x = F.leaky_relu(x)
        x = self.decA2(x)

        x = self.third_deconv(x)
        x = self.third_batch_norm(x)
        x = F.leaky_relu(x)
        x = self.decA3(x)

        x = self.fourth_deconv(x)
        x = self.fourth_batch_norm(x)

        x = self.fifth_deconv(x)
        sigmoid_out = nn.functional.sigmoid(x)
        #tanh_out = nn.functional.tanh(x)

        #out = (tanh_out + 1) * 255 / 2
        out = sigmoid_out * 255
        
        # print 'out.shape =', out.shape

        return out


class Discriminator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Discriminator, self).__init__()

        # input image will have the size of 64x64x3
        self.first_conv_layer = TransitionDown(in_channels=3, out_channels=128, kernel_size=3)
        self.second_conv_layer = TransitionDown(in_channels=128, out_channels=256, kernel_size=3)
        self.third_conv_layer = TransitionDown(in_channels=256, out_channels=512, kernel_size=3)
        self.fourth_conv_layer = TransitionDown(in_channels=512, out_channels=512, kernel_size=3)

        self.fc1 = nn.Linear(4 * 4 * 512, 100)
        self.fc2 = nn.Linear(100, 1)

        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)

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

        x = x.view(-1, 4 * 4 * 512)
        x = F.relu(self.fc1(x))
        
        # disable relu at the last layer
        x = self.fc2(x)

        # disable sigmoid for wasserstein gan
        #out = nn.functional.sigmoid(x)
        
        out = x

        return out

################################################################################
# Modules borrowed from DRIT

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

###################################################################################

if __name__ == "__main__":
    print 'main'

    gen_model = Generator()
    disc_model = Discriminator()

    if is_gpu_mode:
        gen_model.cuda()
        disc_model.cuda()
        # gen_model = torch.nn.DataParallel(gen_model).cuda()
        # disc_model = torch.nn.DataParallel(disc_model).cuda()

    '''
    optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE_GENERATOR)
    optimizer_disc = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)
    '''
    
    # use RMSprop instead of Adam for WGAN
    optimizer_gen = torch.optim.RMSprop(gen_model.parameters(), lr=LEARNING_RATE_GENERATOR)
    optimizer_disc = torch.optim.RMSprop(disc_model.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)

    # read imgs
    image_buff_read_index = 0

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    # answer_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))

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

        if is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
            noise_z = Variable(noise_z.cuda())
        else:
            inputs = Variable(torch.from_numpy(input_img).float())
            noise_z = Variable(noise_z)

        # feedforward the inputs. generator
        outputs_gen = gen_model(noise_z)

        # feedforward the inputs. discriminator
        output_disc_real = disc_model(inputs)
        output_disc_fake = disc_model(outputs_gen)

        # loss functions (vanilla gan)
        '''
        loss_real_d = F.binary_cross_entropy(output_disc_real, ones_label)
        loss_fake_d = F.binary_cross_entropy(output_disc_fake, zeros_label)
        loss_disc_total = loss_real_d + loss_fake_d

        loss_gen = F.binary_cross_entropy(output_disc_fake, ones_label)

        # loss_disc_total = -torch.mean(torch.log(output_disc_real) + torch.log(1. - output_disc_fake))
        # loss_gen = -torch.mean(torch.log(output_disc_fake))
        '''

        # lsgan loss for the discriminator
        '''
        loss_disc_total = 0.5 * (torch.mean((output_disc_real - 1) ** 2) + torch.mean(output_disc_fake ** 2))

        # lsgan loss for the generator
        loss_gen = 0.5 * torch.mean((output_disc_fake - 1) ** 2)
        '''
        
        # wasserstein gan loss 
        loss_disc_total = -(torch.mean(output_disc_real) - torch.mean(output_disc_fake))
        loss_gen = -torch.mean(output_disc_fake)
        
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer_disc.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_disc_total.backward(retain_graph=True)

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer_disc.step()

        # weight clamping
        # (ref: https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/)
        for param in disc_model.parameters():
            param.data.clamp_(-0.01, 0.01)
    
        # generator
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

        if i % 100 == 0:
            print '-----------------------------------------------'
            print '-----------------------------------------------'
            print 'iterations = ', str(i)
            print 'loss(generator)     = ', str(loss_gen)
            print 'loss(discriminator) = ', str(loss_disc_total)
            print '-----------------------------------------------'
            print '(discriminator out-real) = ', output_disc_real[0]
            print '(discriminator out-fake) = ', output_disc_fake[0]

            # tf-board (scalar)
            logger.scalar_summary('loss-generator', loss_gen, i)
            logger.scalar_summary('loss-discriminator', loss_disc_total, i)
            # logger.scalar_summary('disc-out-for-real', output_disc_real[0], i)
            # logger.scalar_summary('disc-out-for-fake', output_disc_fake[0], i)

            # tf-board (images - first 10 batches)
            output_imgs_temp = outputs_gen.cpu().data.numpy()[0:6]
            input_imgs_temp = inputs.cpu().data.numpy()[0:4]
            # logger.an_image_summary('generated', output_img, i)

            # rgb to bgr
            output_imgs_temp = output_imgs_temp[:, [2, 1, 0], ...]
            input_imgs_temp = input_imgs_temp[:, [2, 1, 0], ...]

            logger.image_summary('generated', output_imgs_temp, i)
            logger.image_summary('real', input_imgs_temp, i)

        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(gen_model.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'wgan_gp_pytorch_iter_' + str(i) + '.pt')