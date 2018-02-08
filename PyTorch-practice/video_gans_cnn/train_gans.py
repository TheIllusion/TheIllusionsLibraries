import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time, os
import numpy as np
import data_loader
from logger import Logger
from generator_tiramisu import Tiramisu

print 'train_gans.py'

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
MODEL_SAVING_FREQUENCY = 10000

# tbt005 (10.161.31.83)
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/models/'
RESULT_IMAGE_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/result_imgs/'
TENSORBOARD_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/tf_board_logger/'

# tensor-board logger
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

class Discriminator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Discriminator, self).__init__()

        # input image will have the size of 64x64x3
        self.first_conv_layer = TransitionDown(in_channels=6, out_channels=32, kernel_size=3)
        self.second_conv_layer = TransitionDown(in_channels=32, out_channels=64, kernel_size=3)
        self.third_conv_layer = TransitionDown(in_channels=64, out_channels=128, kernel_size=3)
        self.fourth_conv_layer = TransitionDown(in_channels=128, out_channels=256, kernel_size=3)
        self.fifth_conv_layer = TransitionDown(in_channels=256, out_channels=512, kernel_size=3)

        self.fc1 = nn.Linear(8 * 8 * 512, 10)
        self.fc2 = nn.Linear(10, 1)

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

        x = x.view(BATCH_SIZE, 8 * 8 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        sigmoid_out = nn.functional.sigmoid(x)

        return sigmoid_out


###################################################################################

if is_gpu_mode:
    ones_label = Variable(torch.ones(BATCH_SIZE).cuda())
    zeros_label = Variable(torch.zeros(BATCH_SIZE).cuda())
else:
    ones_label = Variable(torch.ones(BATCH_SIZE))
    zeros_label = Variable(torch.zeros(BATCH_SIZE))

if __name__ == "__main__":
    print 'main'

    gen_model = Tiramisu()
    disc_model = Discriminator()

    if is_gpu_mode:
        gen_model.cuda()
        disc_model.cuda()
        # gen_model = torch.nn.DataParallel(gen_model).cuda()
        # disc_model = torch.nn.DataParallel(disc_model).cuda()

    optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE_GENERATOR)
    optimizer_disc = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)

    # read imgs
    image_buff_read_index = 0

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    answer_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))

    # opencv style
    output_img_opencv = np.empty(shape=(data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT, 3))

    if not os.path.exists(RESULT_IMAGE_DIRECTORY):
        os.mkdir(RESULT_IMAGE_DIRECTORY)

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

        # random noise z
        # noise_z is no longer necessary for pix2pix
        #noise_z = torch.randn(BATCH_SIZE, 1, 4, 4)

        if is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
            answers = Variable(torch.from_numpy(answer_img).float().cuda())
            #noise_z = Variable(noise_z.cuda())
        else:
            inputs = Variable(torch.from_numpy(input_img).float())
            answers = Variable(torch.from_numpy(answer_img).float())
            #noise_z = Variable(noise_z)

        # feedforward the inputs. generator
        #outputs_gen = gen_model(noise_z)
        outputs_gen = gen_model(inputs)

        # feedforward the (input, answer) pairs. discriminator
        output_disc_real = disc_model(torch.cat((inputs, answers), 1))
        output_disc_fake = disc_model(torch.cat((inputs, outputs_gen), 1))

        # loss functions
        # vanilla gan loss for the discriminator
        '''
        loss_real_d = F.binary_cross_entropy(output_disc_real, ones_label)
        loss_fake_d = F.binary_cross_entropy(output_disc_fake, zeros_label)
        loss_disc_total = loss_real_d + loss_fake_d
        '''

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
        optimizer_gen.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_disc_total_lsgan.backward(retain_graph=True)

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer_disc.step()

        # generator
        optimizer_disc.zero_grad()
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
            # logger.an_image_summary('generated', output_img, i)
            logger.image_summary('generated', output_imgs_temp, i)
            logger.image_summary('real', answer_imgs_temp, i)

        if i % 500 == 0:
            # save the output images
            # feedforward the inputs. generator

            for file_idx in range(1):
                # random noise z
                '''
                noise_z = torch.randn(BATCH_SIZE, 1, 4, 4)

                if is_gpu_mode:
                    noise_z = Variable(noise_z.cuda())
                else:
                    noise_z = Variable(noise_z)
                '''

                outputs_gen = gen_model(inputs)
                output_img = outputs_gen.cpu().data.numpy()[0]

                output_img_opencv[:, :, 0] = output_img[0, :, :]
                output_img_opencv[:, :, 1] = output_img[1, :, :]
                output_img_opencv[:, :, 2] = output_img[2, :, :]

                cv2.imwrite(os.path.join(RESULT_IMAGE_DIRECTORY, \
                                         'pix2pix_generated_iter_' + str(i) + '_' + str(file_idx) + '.jpg'), output_img_opencv)

        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(gen_model.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'pix2pix_gen_model_iter_' + str(i) + '.pt')
