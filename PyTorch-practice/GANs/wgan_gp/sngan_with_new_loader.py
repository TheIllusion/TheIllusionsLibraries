# using python3 and torchgan (run within docker image)

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time, os
import numpy as np
#import data_loader
from logger import Logger
#import torchgan
from torchgan.layers.spectralnorm import SpectralNorm2d
from torchgan.layers import SelfAttention2d

#from torchgan.models import DCGANGenerator, DCGANDiscriminator
#from torchgan.losses import MinimaxGeneratorLoss, MinimaxDiscriminatorLoss,
#from torchgan.trainer import Trainer
from torch.utils.data import Dataset, DataLoader
import custom_pytorch_dataloader
from custom_pytorch_dataloader import FaceDataset, Rescale, RandomCrop, ToTensor
from torchvision import transforms

#TEST_NAME = 'sngan_with_batch_norm_2'
#TEST_NAME = 'sngan_light_weight_gen_temp_larger_set'
#TEST_NAME = 'sngan_tuned'
#TEST_NAME = 'sngan_tuned_with_scheduler'
#TEST_NAME = 'sagan_both_gen_and_disc'
#TEST_NAME = 'sngan_network_tuned_2'
TEST_NAME = 'sagan_network_tuned_2'

INPUT_IMAGE_DIRECTORY_PATH = "/tbt005_home/rklee/tiny_dataset/faces/"
#INPUT_IMAGE_DIRECTORY_PATH = "/tbt005_home/rklee/temp/original_resized_face/"

# gpu mode
is_gpu_mode = True

# multi gpu
device_count = 1
device_ids = range(device_count)
num_workers = 4 * device_count
#num_workers = 8

# batch size for SNGAN
#BATCH_SIZE = 128 * device_count

# batch size for SAGAN
BATCH_SIZE = 64

TOTAL_ITERATION = 100000

# learning rate (seems to be an optimal choice for single alternate training)
#LEARNING_RATE_GENERATOR = 0.5 * 1e-4
#LEARNING_RATE_DISCRIMINATOR = 0.1 * 1e-4

# also successful (the hyper-parameter suggested in TTUR paper)
#LEARNING_RATE_GENERATOR = 3 * 1e-4
#LEARNING_RATE_DISCRIMINATOR = 1 * 1e-4

# reliable settings
#LEARNING_RATE_GENERATOR = 3 * 1e-4
#LEARNING_RATE_DISCRIMINATOR = 0.5 * 1e-4

# SNGAN
#LEARNING_RATE_GENERATOR = 2 * 1e-4
#LEARNING_RATE_DISCRIMINATOR = 0.5 * 1e-4

# SNGAN
#LEARNING_RATE_GENERATOR = 1 * 1e-4
#LEARNING_RATE_DISCRIMINATOR = 2 * 1e-4

# learning rate (for multiple critic updates) 
# SAGAN paper: By default, the learning rate for the discriminator is 0.0004 and the learning rate for the generator is 0.0001
LEARNING_RATE_GENERATOR = 1 * 1e-4
LEARNING_RATE_DISCRIMINATOR = 1 * 1e-4
CRITIC_MULTIPLE_UPDATES = 1

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 10000

# t005
MODEL_SAVING_DIRECTORY = "/root/TheIllusionsLibraries/PyTorch-practice/GANs/wgan_gp/checkpoints_" + TEST_NAME + '/'

#TF_BOARD_DIRECTOR = "/root/TheIllusionsLibraries/PyTorch-practice/GANs/wgan_gp/tfboard_" + TEST_NAME + '/'

TF_BOARD_DIRECTOR = '/root/cp-vton/tensorboard/sngan_tanh_' + TEST_NAME + '/'

#TF_BOARD_DIRECTOR = "/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/wgan_gp/tfboard/"

RESULT_IMAGE_DIRECTORY = "/root/TheIllusionsLibraries/PyTorch-practice/GANs/wgan_gp/gen_images_" + TEST_NAME + '/'

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
        #torch.nn.init.xavier_uniform(self.conv.weight)
        torch.nn.init.xavier_uniform(self.conv.weight.data, 1.)
        
        self.conv = SpectralNorm2d(self.conv)
        
        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.maxpool = nn.MaxPool2d((3, 3), stride=1)

    def forward(self, x):
        # x = F.max_pool2d(input=x, kernel_size=2)
        # x = self.drop_out(x)
        x = self.conv(x)
        x = F.relu(x)
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
        #torch.nn.init.xavier_uniform(self.transpoed_conv.weight)
        torch.nn.init.xavier_uniform(self.transpoed_conv.weight.data, 1.)
        
        self.transpoed_conv = SpectralNorm2d(self.transpoed_conv)

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
        
        self.dense = nn.Linear(100, 512 * 4 * 4)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        
        self.dense_batch_norm = nn.InstanceNorm2d(512)

        # output feature map will have the size of 8x8x3
        self.first_deconv = TransitionUp(in_channels=512, out_channels=256, stride=2, kernel_size=4)
        #self.first_batch_norm = nn.BatchNorm2d(256)
        self.first_batch_norm = nn.InstanceNorm2d(256)
        #self.decA1 = INSResBlock(256, 256)
        
        # output feature map will have the size of 16x16x3
        self.second_deconv = TransitionUp(in_channels=256, out_channels=128, stride=2, kernel_size=4)
        self.second_batch_norm = nn.InstanceNorm2d(128)
        #self.decA2 = INSResBlock(128, 128)

        # output feature map will have the size of 32x32x3
        self.third_deconv = TransitionUp(in_channels=128, out_channels=64, stride=2, kernel_size=4)
        self.third_batch_norm = nn.InstanceNorm2d(64)
        
        # self attention layer
        self.self_attention = SelfAttention2d(64)
        
        #self.decA3 = INSResBlock(64, 64)

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
        
        x = self.dense(x)
        x = x.view(-1, 512, 4, 4)
        x = self.dense_batch_norm(x)
        x = F.leaky_relu(x)
        
        x = self.first_deconv(x)
        x = self.first_batch_norm(x)
        x = F.leaky_relu(x)
        #x = self.decA1(x)

        x = self.second_deconv(x)
        x = self.second_batch_norm(x)
        x = F.leaky_relu(x)
        #x = self.decA2(x)

        x = self.third_deconv(x)
        x = self.third_batch_norm(x)
        x = F.leaky_relu(x)
        
        # self-attention layer
        x = self.self_attention(x)
        
        #x = self.decA3(x)

        x = self.fourth_deconv(x)
        x = self.fourth_batch_norm(x)
        x = F.leaky_relu(x)
                        
        x = self.fifth_deconv(x)
        #sigmoid_out = nn.functional.sigmoid(x)
        tanh_out = nn.functional.tanh(x)

        #out = (tanh_out + 1) * 255 / 2
        #out = sigmoid_out
        out = tanh_out
        
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
        
        # self attention layer
        self.self_attention = SelfAttention2d(128)
        
        self.second_conv_layer = TransitionDown(in_channels=128, out_channels=256, kernel_size=3)
        self.third_conv_layer = TransitionDown(in_channels=256, out_channels=256, kernel_size=3)
        self.fourth_conv_layer = TransitionDown(in_channels=256, out_channels=256, kernel_size=3)

        self.fc1 = nn.Linear(4 * 4 * 256, 1)

        torch.nn.init.xavier_uniform(self.fc1.weight.data, 1.)
        
        self.fc1 = SpectralNorm2d(self.fc1)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        x = self.first_conv_layer(x)
        
        # self attention layer
        x = self.self_attention(x)
        
        x = self.second_conv_layer(x)
        x = self.third_conv_layer(x)
        x = self.fourth_conv_layer(x)

        x = x.view(-1, 4 * 4 * 256)
        x = self.fc1(x)
        
        # disable sigmoid for wasserstein gan
        #out = nn.functional.sigmoid(x)
        
        out = x

        return out
    
    # from ganimation
    def calculate_gradient_penalty(self, real_images, fake_images, batch_size):       
        lambda_term = 10
        # interpolate sample
        alpha = torch.rand(batch_size, 1, 1, 1).cuda().expand_as(real_images)
        interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
        interpolated_prob = self.forward(interpolated)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        grad_penalty = torch.mean((grad_l2norm - 1) ** 2) * lambda_term

        return grad_penalty
    
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
    print ('main')

    gen_model = Generator()
    disc_model = Discriminator()
    
    if is_gpu_mode:
        gen_model.cuda()
        disc_model.cuda()
       
        # multi gpu
        device = torch.device("cuda:{:d}".format(device_ids[0] if device_ids else 0))
        gen_model = nn.DataParallel(gen_model, device_ids=device_ids)
        disc_model = nn.DataParallel(disc_model, device_ids=device_ids)
        gen_model.to(device)
        disc_model.to(device)

    optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE_GENERATOR)
    optimizer_disc = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)
    
    # scheduler. use an exponentially decaying learning rate.
    scheduler_gen = torch.optim .lr_scheduler.ExponentialLR(optimizer_gen, gamma=0.99)
    scheduler_disc = torch.optim .lr_scheduler.ExponentialLR(optimizer_disc, gamma=0.99)
   
    ''' 
    # use RMSprop instead of Adam for WGAN
    optimizer_gen = torch.optim.RMSprop(gen_model.parameters(), lr=LEARNING_RATE_GENERATOR)
    optimizer_disc = torch.optim.RMSprop(disc_model.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)
    '''

    # read imgs
    image_buff_read_index = 0
    
    # dataset
    transformed_dataset = FaceDataset(root_dir=INPUT_IMAGE_DIRECTORY_PATH, \
                                  transform=transforms.Compose([
                          Rescale(custom_pytorch_dataloader.INPUT_IMAGE_WIDTH+5),
                          RandomCrop(custom_pytorch_dataloader.INPUT_IMAGE_WIDTH),
                          ToTensor()]))
    
    # data_loader
    dataloader = DataLoader(transformed_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=num_workers)

    #for i in range(TOTAL_ITERATION):
    
    i = 0
    exit_flag = False
    while True:
        
        if exit_flag == True:
            break
            
        for img_idx, inputs in enumerate(dataloader):
            
            i += 1
            if i % TOTAL_ITERATION == 0:
                exit_flag = True
                break

            exit_notification = False

            for critic_iter in range(CRITIC_MULTIPLE_UPDATES):

                # random noise z
                #noise_z = torch.randn(BATCH_SIZE, 3, 4, 4)
                noise_z = torch.randn(BATCH_SIZE, 100)

                inputs = inputs['image']  

                if is_gpu_mode:
                    #inputs = Variable(torch.from_numpy(input_img).float().cuda())
                    inputs = Variable(inputs.float().cuda())
                    noise_z = Variable(noise_z.cuda())
                    # multi gpu
                    inputs = inputs.to(device)
                    noise_z = noise_z.to(device)
                else:
                    #inputs = Variable(torch.from_numpy(input_img).float())
                    noise_z = Variable(noise_z)

                # normalize input range [-1, +1]
                inputs = (inputs - 0.5) * 2.0
                # feedforward the inputs. generator
                outputs_gen = gen_model(noise_z)

                # pseudo zero-center
                '''
                print ('max inputs =', torch.max(inputs))
                print ('max ouputs_gen =', torch.max(outputs_gen))
                print ('min inputs =', torch.min(inputs))
                print ('min ouputs_gen =', torch.min(outputs_gen))
                '''
                
                # feedforward the inputs. discriminator
                output_disc_real = disc_model(inputs)
                output_disc_fake = disc_model(outputs_gen)

                # wasserstein gan hinge loss 
                loss_disc_total = nn.ReLU()(1.0 - output_disc_real).mean() \
                                  + nn.ReLU()(1.0 + output_disc_fake).mean()

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable weights
                # of the model)
                optimizer_disc.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss_disc_total.backward(retain_graph=True)

                optimizer_disc.step()

                # disable for wgan-gp
                '''
                # weight clamping
                # (ref: https://wiseodd.github.io/techblog/2017/02/04/wasserstein-gan/)
                for param in disc_model.parameters():
                    param.data.clamp_(-0.01, 0.01)
                '''

                # gradient penalty
                '''
                grad_penalty = disc_model.calculate_gradient_penalty(inputs, \
                                                                     outputs_gen, \
                                                                     batch_size=BATCH_SIZE)
                optimizer_disc.zero_grad()
                grad_penalty.backward(retain_graph=True)
                optimizer_disc.step()
                '''

            # generator update
            optimizer_gen.zero_grad()        
            # random noise z
            #noise_z = torch.randn(BATCH_SIZE, 3, 4, 4)
            noise_z = torch.randn(BATCH_SIZE, 100)

            if is_gpu_mode:
                noise_z = Variable(noise_z.cuda())
            else:
                noise_z = Variable(noise_z)

            # feedforward the inputs. generator
            outputs_gen = gen_model(noise_z)   
            
            '''
            print ('torch.max(outputs_gen) =', torch.max(outputs_gen))
            print ('torch.min(outputs_gen) =', torch.min(outputs_gen))
            print ('torch.mean(outputs_gen) =', torch.mean(outputs_gen))
            '''
            
            output_disc_fake = disc_model(outputs_gen)
            loss_gen = -torch.mean(output_disc_fake)
            loss_gen.backward()
            optimizer_gen.step()

            if i % 30 == 0:
                print ('-----------------------------------------------')
                print ('-----------------------------------------------')
                print ('iterations = ', str(i))
                print ('loss(generator)     = ', str(loss_gen))
                print ('loss(discriminator) = ', str(loss_disc_total))
                #print ('grad_penalty =', grad_penalty)
                print ('-----------------------------------------------')
                print ('(discriminator out-real) = ', output_disc_real[0:4])
                print ('(discriminator out-fake) = ', output_disc_fake[0:4])

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
                #output_imgs_temp = output_imgs_temp[:, [2, 1, 0], ...]
                #input_imgs_temp = input_imgs_temp[:, [2, 1, 0], ...]

                logger.image_summary('generated', output_imgs_temp, i)
                logger.image_summary('real', input_imgs_temp, i)

            # save the model
            if i % MODEL_SAVING_FREQUENCY == 0:
                torch.save(gen_model.state_dict(),
                           MODEL_SAVING_DIRECTORY + '_iter_' + str(i) + '.pt')

