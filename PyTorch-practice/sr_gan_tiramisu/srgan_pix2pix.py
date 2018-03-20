import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time, os
import numpy as np
import data_loader_for_pix2pix as data_loader
from logger import Logger
from generator_modified_tiramisu import Tiramisu

print 'srgan_pix2pix.py'

# gpu mode
is_gpu_mode = True

# batch size
BATCH_SIZE = 1
TOTAL_ITERATION = 1000000

# learning rate
LEARNING_RATE_GENERATOR = 3 * 1e-4
LEARNING_RATE_DISCRIMINATOR = 1 * 1e-4

# zero centered
# MEAN_VALUE_FOR_ZERO_CENTERED = 128

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 2000

# T005
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/sr_gan_tiramisu/generator_checkpoints/'

TF_BOARD_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/sr_gan_tiramisu/tfboard/'

if not os.path.exists(MODEL_SAVING_DIRECTORY):
    os.mkdir(MODEL_SAVING_DIRECTORY)

if not os.path.exists(TF_BOARD_DIRECTORY):
    os.mkdir(TF_BOARD_DIRECTORY)

# tensor-board logger
logger = Logger(TF_BOARD_DIRECTORY)

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TransitionDown, self).__init__()

        # self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, padding=1, stride=2, bias=False)

        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x = F.max_pool2d(input=x, kernel_size=2)
        # x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Discriminator, self).__init__()

        # input image will have the size of 1024x1024x3
        self.first_conv_layer = TransitionDown(in_channels=3, out_channels=16, kernel_size=3)
        # 512x512
        self.second_conv_layer = TransitionDown(in_channels=16, out_channels=32, kernel_size=3)
        # 256x256
        self.third_conv_layer = TransitionDown(in_channels=32, out_channels=64, kernel_size=3)
        # 128x128
        self.fourth_conv_layer = TransitionDown(in_channels=64, out_channels=128, kernel_size=3)
        # 64x64
        self.fifth_conv_layer = TransitionDown(in_channels=128, out_channels=256, kernel_size=3)
        # 32x32
        self.last_conv_layer = TransitionDown(in_channels=256, out_channels=512, kernel_size=3)
        
        self.fc1 = nn.Linear(16 * 16 * 512, 2)
        self.fc2 = nn.Linear(2, 1)

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
        x = self.fifth_conv_layer(x)
        x = self.last_conv_layer(x)
        
        #x = x.view(BATCH_SIZE, 8 * 8 * 512)
        x = x.view(-1, 16 * 16 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        sigmoid_out = nn.functional.sigmoid(x)

        return sigmoid_out


###################################################################################

if __name__ == "__main__":
    print 'main'

    gen_model = Tiramisu()
    disc_model = Discriminator()

    if is_gpu_mode:
        gen_model.cuda()
        disc_model.cuda()

    optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE_GENERATOR)
    optimizer_disc = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)

    # read imgs
    image_buff_read_index = 0

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    answer_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH_ANSWER, data_loader.INPUT_IMAGE_HEIGHT_ANSWER))

    # opencv style
    output_img_opencv = np.empty(shape=(data_loader.INPUT_IMAGE_WIDTH_ANSWER, data_loader.INPUT_IMAGE_HEIGHT_ANSWER, 3))

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
            np.copyto(answer_img[j], data_loader.answer_buff[image_buff_read_index])

            data_loader.buff_status[image_buff_read_index] = 'empty'

            image_buff_read_index = image_buff_read_index + 1
            if image_buff_read_index >= data_loader.image_buffer_size:
                image_buff_read_index = 0

        if is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
            answers = Variable(torch.from_numpy(answer_img).float().cuda())
        else:
            inputs = Variable(torch.from_numpy(input_img).float())
            answers = Variable(torch.from_numpy(answer_img).float())

        # feedforward the inputs. generator
        #outputs_gen = gen_model(noise_z)
        outputs_gen = gen_model(inputs)

        # feedforward the (input, answer) pairs. discriminator
        output_disc_real = disc_model(answers)
        output_disc_fake = disc_model(outputs_gen)

        # lsgan loss for the discriminator
        loss_disc_total_lsgan = 0.5 * (torch.mean((output_disc_real - 1)**2) + torch.mean(output_disc_fake**2))

        # l1-loss between real and fake
        l1_loss = F.l1_loss(outputs_gen, answers)

        # vanilla gan loss for the generator
        #loss_gen_vanilla = F.binary_cross_entropy(output_disc_fake, ones_label)

        # lsgan loss for the generator
        loss_gen_lsgan = 0.5 * torch.mean((output_disc_fake - 1)**2)

        loss_gen_total_lsgan = 5 * loss_gen_lsgan + 0.01 * l1_loss

        # loss_disc_total = -torch.mean(torch.log(output_disc_real) + torch.log(1. - output_disc_fake))
        # loss_gen = -torch.mean(torch.log(output_disc_fake))

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer_disc.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_disc_total_lsgan.backward(retain_graph=True)

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer_disc.step()

        # generator
        optimizer_gen.zero_grad()
        loss_gen_total_lsgan.backward()
        optimizer_gen.step()

        if i % 50 == 0:
            print '-----------------------------------------------'
            print '-----------------------------------------------'
            print 'iterations = ', str(i)
            print 'loss(generator)     = ', str(loss_gen_lsgan)
            print 'loss(l1)     = ', str(l1_loss)
            print 'loss(discriminator) = ', str(loss_disc_total_lsgan)
            print '-----------------------------------------------'
            print '(discriminator out-real) = ', output_disc_real[0]
            print '(discriminator out-fake) = ', output_disc_fake[0]

            # tf-board (scalar)
            logger.scalar_summary('loss-generator', loss_gen_lsgan, i)
            logger.scalar_summary('loss-l1', l1_loss, i)
            logger.scalar_summary('loss-discriminator', loss_disc_total_lsgan, i)
            logger.scalar_summary('disc-out-for-real', output_disc_real[0], i)
            logger.scalar_summary('disc-out-for-fake', output_disc_fake[0], i)

            # tf-board (images - first 1 batches)
            output_imgs_temp = outputs_gen.cpu().data.numpy()[0:1]
            answer_imgs_temp = answers.cpu().data.numpy()[0:1]
            inputs_temp = inputs.cpu().data.numpy()[0:1]
            # logger.an_image_summary('generated', output_img, i)
            logger.image_summary('generated', output_imgs_temp, i)
            logger.image_summary('real', answer_imgs_temp, i)
            logger.image_summary('input', inputs_temp, i)
            logger.image_summary('input(dup)', inputs_temp, i)

        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(gen_model.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'sr_gan_tiramisu_iter_' + str(i) + '.pt')
