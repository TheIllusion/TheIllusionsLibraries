import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time, os
import numpy as np
import itertools
import data_loader_for_cyclegan as data_loader
from logger import Logger
from unet_model import Unet

print 'simple_cyclegan_unet.py'

# gpu mode
is_gpu_mode = True

# batch size
BATCH_SIZE = 2
TOTAL_ITERATION = 1000000

# learning rate
LEARNING_RATE_GENERATOR = 3 * 1e-4
LEARNING_RATE_DISCRIMINATOR = 1 * 1e-4

# zero centered
# MEAN_VALUE_FOR_ZERO_CENTERED = 128

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 10000

# transfer learning option
ENABLE_TRANSFER_LEARNING = False

# tbt005
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/models_unet/'
RESULT_IMAGE_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/generate_imgs_simple_cyclegan_unet/'

# i7-2600k
#MODEL_SAVING_DIRECTORY = '/home/illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/GANs/models/'
#RESULT_IMAGE_DIRECTORY = '/home/illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/GANs/generate_imgs_simple_cyclegan/'

# tensor-board logger
if not os.path.exists(MODEL_SAVING_DIRECTORY + 'tf_board_logger'):
    os.mkdir(MODEL_SAVING_DIRECTORY + 'tf_board_logger')

logger = Logger(MODEL_SAVING_DIRECTORY + 'tf_board_logger')

###############################################################################
# Discriminator Network

class ConvolutionDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvolutionDown, self).__init__()

        # self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, padding=1, stride=1, bias=True)

        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x = F.max_pool2d(input=x, kernel_size=2)
        # x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)
        
        # added
        x = F.max_pool2d(input=x, kernel_size=2)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Discriminator, self).__init__()

        # input image will have the size of 64x64x3
        self.first_conv_layer = ConvolutionDown(in_channels=3, out_channels=32, kernel_size=3)
        self.second_conv_layer = ConvolutionDown(in_channels=32, out_channels=64, kernel_size=3)
        self.third_conv_layer = ConvolutionDown(in_channels=64, out_channels=128, kernel_size=3)
        self.fourth_conv_layer = ConvolutionDown(in_channels=128, out_channels=256, kernel_size=3)
        self.fifth_conv_layer = ConvolutionDown(in_channels=256, out_channels=512, kernel_size=3)
        self.last_conv_layer = ConvolutionDown(in_channels=512, out_channels=1, kernel_size=3)
        

        #self.fc1 = nn.Linear(8 * 8 * 512, 5)
        #self.fc2 = nn.Linear(5, 1)

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
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))

        sigmoid_out = nn.functional.sigmoid(x)

        return sigmoid_out

###############################################################################

if is_gpu_mode:
    ones_label = Variable(torch.ones(BATCH_SIZE).cuda())
    zeros_label = Variable(torch.zeros(BATCH_SIZE).cuda())
else:
    ones_label = Variable(torch.ones(BATCH_SIZE))
    zeros_label = Variable(torch.zeros(BATCH_SIZE))

if __name__ == "__main__":
    print 'main'

    gen_model_a = Unet()
    gen_model_b = Unet()
    disc_model_a = Discriminator()
    disc_model_b = Discriminator()

    if is_gpu_mode:
        gen_model_a.cuda()
        gen_model_b.cuda()
        disc_model_a.cuda()
        disc_model_b.cuda()
        # gen_model = torch.nn.DataParallel(gen_model).cuda()
        # disc_model = torch.nn.DataParallel(disc_model).cuda()

    if ENABLE_TRANSFER_LEARNING:
        # load the saved checkpoints for hair semantic segmentation
        gen_model_a.load_state_dict(torch.load('/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/tiramisu-fcdensenet103/models/tiramisu_lfw_added_zero_centr_lr_0_0002_iter_1870000.pt'))
        gen_model_b.load_state_dict(torch.load('/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/tiramisu-fcdensenet103/models/tiramisu_lfw_added_zero_centr_lr_0_0002_iter_1870000.pt'))
        
    gen_params_total = itertools.chain(gen_model_a.parameters(), gen_model_b.parameters())
    optimizer_gen = torch.optim.Adam(gen_params_total, lr=LEARNING_RATE_GENERATOR)

    optimizer_disc_a = torch.optim.Adam(disc_model_a.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)
    optimizer_disc_b = torch.optim.Adam(disc_model_b.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)

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

        # feedforward the inputs. generators.
        outputs_gen_a_to_b = gen_model_a(inputs)
        outputs_gen_b_to_a = gen_model_a(answers)

        # feedforward the data to the discriminator_a
        output_disc_real_a = disc_model_a(inputs)
        output_disc_fake_a = disc_model_a(outputs_gen_b_to_a)

        # feedforward the data to the discriminator_b
        output_disc_real_b = disc_model_b(answers)
        output_disc_fake_b = disc_model_b(outputs_gen_a_to_b)

        # loss functions

        # lsgan loss for the discriminator_a
        loss_disc_a_lsgan = 0.5 * (torch.mean((output_disc_real_a - 1)**2) + torch.mean(output_disc_fake_a**2))

        # lsgan loss for the discriminator_b
        loss_disc_b_lsgan = 0.5 * (torch.mean((output_disc_real_b - 1) ** 2) + torch.mean(output_disc_fake_b ** 2))

        # cycle-consistency loss(a)
        reconstructed_a = gen_model_b(outputs_gen_a_to_b)
        l1_loss_rec_a = F.l1_loss(reconstructed_a, inputs)

        # cycle-consistency loss(b)
        reconstructed_b = gen_model_a(outputs_gen_b_to_a)
        l1_loss_rec_b = F.l1_loss(reconstructed_b, answers)

        # lsgan loss for the generator_a
        loss_gen_lsgan_a = 0.5 * torch.mean((output_disc_fake_b - 1)**2)

        # lsgan loss for the generator_b
        loss_gen_lsgan_b = 0.5 * torch.mean((output_disc_fake_a - 1) ** 2)

        loss_gen_total_lsgan = loss_gen_lsgan_a + loss_gen_lsgan_b + 0.01 * (l1_loss_rec_a + l1_loss_rec_b)

        # discriminator_a
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer_disc_a.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_disc_a_lsgan.backward(retain_graph=True)
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer_disc_a.step()

        # discriminator_b
        optimizer_disc_b.zero_grad()
        loss_disc_b_lsgan.backward(retain_graph=True)
        optimizer_disc_b.step()

        # generators
        optimizer_gen.zero_grad()
        loss_gen_total_lsgan.backward()
        optimizer_gen.step()

        if i % 30 == 0:
            print '-----------------------------------------------'
            print '-----------------------------------------------'
            print 'iterations = ', str(i)
            print 'loss(generator_a)     = ', str(loss_gen_lsgan_a)
            print 'loss(generator_b)     = ', str(loss_gen_lsgan_b)
            print 'loss(rec_a_l1)     = ', str(l1_loss_rec_a)
            print 'loss(rec_b_l1)     = ', str(l1_loss_rec_b)
            print 'loss(discriminator_a) = ', str(loss_disc_a_lsgan)
            print 'loss(discriminator_b) = ', str(loss_disc_b_lsgan)
            print '-----------------------------------------------'
            print '(discriminator_a out-real) = ', output_disc_real_a[0]
            print '(discriminator_a out-fake) = ', output_disc_fake_a[0]
            print '(discriminator_b out-real) = ', output_disc_real_b[0]
            print '(discriminator_b out-fake) = ', output_disc_fake_b[0]

            # tf-board (scalar)
            logger.scalar_summary('loss(generator_a)', loss_gen_lsgan_a, i)
            logger.scalar_summary('loss(generator_b)', loss_gen_lsgan_b, i)
            logger.scalar_summary('loss-rec_a-l1', l1_loss_rec_a, i)
            logger.scalar_summary('loss-rec_b-l1', l1_loss_rec_b, i)
            logger.scalar_summary('loss(discriminator_a)', loss_disc_a_lsgan, i)
            logger.scalar_summary('loss(discriminator_b)', loss_disc_b_lsgan, i)
            #logger.scalar_summary('disc_a-out-for-real', output_disc_real_a[0], i)
            #logger.scalar_summary('disc_a-out-for-fake', output_disc_fake_a[0], i)
            #logger.scalar_summary('disc_b-out-for-real', output_disc_real_b[0], i)
            #logger.scalar_summary('disc_b-out-for-fake', output_disc_fake_b[0], i)

            # tf-board (images - first 1 batches)
            inputs_imgs_temp = inputs.cpu().data.numpy()[0:1]
            output_imgs_temp = outputs_gen_a_to_b.cpu().data.numpy()[0:1]
            answer_imgs_temp = answers.cpu().data.numpy()[0:1]
            reconstructed_a_temp = reconstructed_a.cpu().data.numpy()[0:1]
            reconstructed_b_temp = reconstructed_b.cpu().data.numpy()[0:1]
            
            # logger.an_image_summary('generated', output_img, i)
            logger.image_summary('input', inputs_imgs_temp, i)
            logger.image_summary('generated', output_imgs_temp, i)
            logger.image_summary('reconstructed_a', reconstructed_a_temp, i)
            logger.image_summary('reconstructed_b', reconstructed_b_temp, i)
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

                outputs_gen = gen_model_a(inputs)
                output_img = outputs_gen.cpu().data.numpy()[0]

                output_img_opencv[:, :, 0] = output_img[0, :, :]
                output_img_opencv[:, :, 1] = output_img[1, :, :]
                output_img_opencv[:, :, 2] = output_img[2, :, :]

                output_img_opencv = output_img_opencv[..., [2,1,0]]
                cv2.imwrite(os.path.join(RESULT_IMAGE_DIRECTORY, \
                                         'cyclegan_unet_generated_iter_' + str(i) + '_' + str(file_idx) + '.jpg'), output_img_opencv)

        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(gen_model_a.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'cyclegan_unet_model_iter_' + str(i) + '.pt')
