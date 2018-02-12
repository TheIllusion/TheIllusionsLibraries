import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time, os
import numpy as np
import data_loader
from logger import Logger
from generator_tiramisu import Tiramisu
from discriminator import Discriminator, BATCH_SIZE

print 'train_gans.py'

# gpu mode
is_gpu_mode = True

# batch size
#BATCH_SIZE = 1
TOTAL_ITERATION = 1000000

# learning rate
LEARNING_RATE_GENERATOR = 2 * 1e-4
LEARNING_RATE_DISCRIMINATOR = 2 * 1e-4

# zero centered
# MEAN_VALUE_FOR_ZERO_CENTERED = 128

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 10000

# tbt005 (10.161.31.83)
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/models_3/'
RESULT_IMAGE_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/result_imgs/'
TENSORBOARD_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/tf_board_logger_3/'

# single test face image
SINGLE_TEST_FACE_IMAGE = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/rk_face.jpg'

test_face_img = cv2.imread(SINGLE_TEST_FACE_IMAGE, cv2.IMREAD_UNCHANGED)
test_face_img = cv2.resize(test_face_img, (data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
test_face_img = test_face_img[..., [2,1,0]]

# tensor-board logger
if not os.path.exists(TENSORBOARD_DIRECTORY):
    os.mkdir(TENSORBOARD_DIRECTORY)

logger = Logger(TENSORBOARD_DIRECTORY)

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
    
    motion_vec_img = np.empty(shape=(BATCH_SIZE, 1, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    
    fake_motion_vec_img = np.empty(shape=(BATCH_SIZE, 1, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    
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
            np.copyto(motion_vec_img[j], data_loader.motion_vector_buff[image_buff_read_index])
            np.copyto(fake_motion_vec_img[j], data_loader.fake_motion_vector_buff[image_buff_read_index])

            data_loader.buff_status[image_buff_read_index] = 'empty'

            image_buff_read_index = image_buff_read_index + 1
            if image_buff_read_index >= data_loader.image_buffer_size:
                image_buff_read_index = 0
            
        if is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
            answers = Variable(torch.from_numpy(answer_img).float().cuda())
            motion_vec = Variable(torch.from_numpy(motion_vec_img).float().cuda())
            fake_motion_vec = Variable(torch.from_numpy(fake_motion_vec_img).float().cuda())
            
        else:
            inputs = Variable(torch.from_numpy(input_img).float())
            answers = Variable(torch.from_numpy(answer_img).float())
            motion_vec = Variable(torch.from_numpy(motion_vec_img).float())
            fake_motion_vec = Variable(torch.from_numpy(fake_motion_vec_img).float())

        # concatenate the motion vector
        inputs_with_mv = torch.cat((motion_vec, inputs), 1)
        inputs_with_fake_mv = torch.cat((fake_motion_vec, inputs), 1)
        
        # feedforward the inputs. generator
        outputs_gen = gen_model(inputs_with_mv)

        # feedforward the (input, answer) pairs. discriminator
        output_disc_real = disc_model(torch.cat((inputs_with_mv, answers), 1))
        output_disc_real_with_fake_vec = disc_model(torch.cat((inputs_with_fake_mv, answers), 1))
        output_disc_fake = disc_model(torch.cat((inputs_with_mv, outputs_gen), 1))
        output_disc_fake_with_fake_vec = disc_model(torch.cat((inputs_with_fake_mv, outputs_gen), 1))

        # loss functions

        # lsgan loss for the discriminator
        loss_disc_total_lsgan = 0.5 * (torch.mean((output_disc_real - 1)**2) + torch.mean(output_disc_fake**2) + 2 * torch.mean(output_disc_real_with_fake_vec**2) + 2 * torch.mean(output_disc_fake_with_fake_vec**2))

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

                inputs_with_mv = torch.cat((inputs, motion_vec), 1)
                
                motion_vec[:,:,:] = 10
                
                outputs_gen = gen_model(inputs_with_mv)
                
                output_img = outputs_gen.cpu().data.numpy()[0]

                output_img_opencv[:, :, 0] = output_img[0, :, :]
                output_img_opencv[:, :, 1] = output_img[1, :, :]
                output_img_opencv[:, :, 2] = output_img[2, :, :]

                cv2.imwrite(os.path.join(RESULT_IMAGE_DIRECTORY, \
                                         'video_gans_generated_iter_' + str(i) + '_' + str(file_idx) + '.jpg'), output_img_opencv)
                
        if i % 200 == 0:
            print '-----------------------------------------------'
            print '-----------------------------------------------'
            print 'iterations = ', str(i)
            print 'loss(generator)     = ', str(loss_gen_lsgan)
            print 'loss(l1)     = ', str(l1_loss)
            print 'loss(discriminator) = ', str(loss_disc_total_lsgan)
            print '-----------------------------------------------'
            print '(discriminator out-real)   = ', output_disc_real[0]
            print '(discriminator out-fake)   = ', output_disc_fake[0]
            print '(discriminator out-real-with-fake-v) = ', output_disc_real_with_fake_vec[0]
            print '(discriminator out-fake-with-fake-v) = ', output_disc_fake_with_fake_vec[0]

            # tf-board (scalar)
            logger.scalar_summary('loss-generator', loss_gen_lsgan, i)
            logger.scalar_summary('loss-l1', l1_loss, i)
            logger.scalar_summary('loss-discriminator', loss_disc_total_lsgan, i)
            logger.scalar_summary('disc-out-for-real', output_disc_real[0], i)
            logger.scalar_summary('disc-out-for-fake', output_disc_fake[0], i)
            logger.scalar_summary('output_disc_real_with_fake_vec', output_disc_real_with_fake_vec[0], i)
            logger.scalar_summary('output_disc_fake_with_fake_vec', output_disc_fake_with_fake_vec[0], i)
            
            # tf-board (images - first 1 batches)
            input_imgs_temp = inputs.cpu().data.numpy()[0:1]
            logger.image_summary('first_batch_input', input_imgs_temp, i)
            
            output_imgs_temp = outputs_gen.cpu().data.numpy()[0:1]
            logger.image_summary('first_batch_generate', output_imgs_temp, i)
                
            answer_imgs_temp = answers.cpu().data.numpy()[0:1]
            logger.image_summary('first_batch_real', answer_imgs_temp, i)
            
            # tf-board (images - generated from test image)
            input_img[0][0, :, :] = test_face_img[:, :, 0]
            input_img[0][1, :, :] = test_face_img[:, :, 1]
            input_img[0][2, :, :] = test_face_img[:, :, 2]
            
            if is_gpu_mode:
                inputs = Variable(torch.from_numpy(input_img).float().cuda())
            else:
                inputs = Variable(torch.from_numpy(input_img).float())
            
            for loop_idx in range(1,21):
                mv_idx = loop_idx * 10
                motion_vec[:,:,:] = mv_idx
                
                # concat
                inputs_with_mv = torch.cat((inputs, motion_vec), 1)
                outputs_gen = gen_model(inputs_with_mv)
                output_imgs_temp = outputs_gen.cpu().data.numpy()[0:1]
                logger.image_summary('2x_cost_generated_from_test_img_mv_idx_' + str(mv_idx), output_imgs_temp, i)
                

        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(gen_model.state_dict(),
                       MODEL_SAVING_DIRECTORY + '2x_cost_video_gans_genenerator_iter_' + str(i) + '.pt')
