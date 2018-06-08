import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time, os
import numpy as np
import itertools
import data_loader_for_unified_cyclegan as data_loader
from logger import Logger
from generator_network_tiramisu import Tiramisu
from discriminator_network import Discriminator, BATCH_SIZE
from data_loader_for_unified_cyclegan import hair_color_list, answer_buff_dict

print 'unified_cyclegan_for_hair_dyeing with acgan (auxiliary classifier GANs)'

# gpu mode
is_gpu_mode = True

TOTAL_ITERATION = 100000

# learning rate
LEARNING_RATE_GENERATOR = 3 * 1e-4
LEARNING_RATE_DISCRIMINATOR = 1 * 1e-4

# zero centered
# MEAN_VALUE_FOR_ZERO_CENTERED = 128

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 5000

# tbt005
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/cyclegan_for_unified_hair_dyeing_acgan/checkpoints_batch_size_1/'

TF_BOARD_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/cyclegan_for_unified_hair_dyeing_acgan/tfboard_batch_size_1/'

# tensor-board logger
if not os.path.exists(TF_BOARD_DIRECTORY):
    os.mkdir(TF_BOARD_DIRECTORY)

logger = Logger(TF_BOARD_DIRECTORY)

###################################################################################

if __name__ == "__main__":
    print 'main'

    gen_model_a = Tiramisu()
    gen_model_b = Tiramisu()
    disc_model_a = Discriminator()
    disc_model_b = Discriminator()

    # cross entropy loss
    criterion_cross_ent = torch.nn.BCELoss()
    
    if is_gpu_mode:
        gen_model_a.cuda()
        gen_model_b.cuda()
        disc_model_a.cuda()
        disc_model_b.cuda()
        '''
        gen_model_a = torch.nn.DataParallel(gen_model_a).cuda()
        gen_model_b = torch.nn.DataParallel(gen_model_b).cuda()
        disc_model_a = torch.nn.DataParallel(disc_model_a).cuda()
        disc_model_b = torch.nn.DataParallel(disc_model_b).cuda()
        '''
        
    gen_params_total = itertools.chain(gen_model_a.parameters(), gen_model_b.parameters())
    optimizer_gen = torch.optim.Adam(gen_params_total, lr=LEARNING_RATE_GENERATOR)

    optimizer_disc_a = torch.optim.Adam(disc_model_a.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)
    optimizer_disc_b = torch.optim.Adam(disc_model_b.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)

    # read imgs
    image_buff_read_index = 0
    saved_image_buff_read_index = 0

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))

    answer_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    
    # conditional vectors
    condition_vectors = np.zeros(shape=(BATCH_SIZE, len(hair_color_list), data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))

    if is_gpu_mode:
        condition_vectors = Variable(torch.from_numpy(condition_vectors).float().cuda())
    else:
        condition_vectors = Variable(torch.from_numpy(condition_vectors).float())
            
    # opencv style
    output_img_opencv = np.empty(shape=(data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT, 3))

    # each iteration
    for i in range(TOTAL_ITERATION):

        exit_notification = False
            
        ############################################################################
        # iterate through colors and manipulate the input data and condition vectors
        for color in hair_color_list:

            # each batch
            image_buff_read_index = saved_image_buff_read_index
            
            for j in range(BATCH_SIZE):

                data_loader.is_main_alive = True

                ####################################################################
                # choose the buffer which is associated with the color
                answer_buff = answer_buff_dict[color]

                while data_loader.buff_status[image_buff_read_index] == 'empty':
                    if exit_notification == True:
                        break

                    time.sleep(1)
                    if data_loader.buff_status[image_buff_read_index] == 'filled':
                        break

                if exit_notification == True:
                    break

                np.copyto(input_img[j], data_loader.input_buff[image_buff_read_index])
                np.copyto(answer_img[j], answer_buff[image_buff_read_index])

                # empty the buffer if the current color is the last color
                if hair_color_list.index(color) == (len(hair_color_list) -1):
                    data_loader.buff_status[image_buff_read_index] = 'empty'

                # debug purposes only
                #print 'color =', color, 'image_buff_read_index =', image_buff_read_index
                
                image_buff_read_index = image_buff_read_index + 1
                if image_buff_read_index >= data_loader.image_buffer_size:
                    image_buff_read_index = 0
           
            ####################################################################
            # manipulate the condition vector
            # set the values of the target plain to 100. and others to 0.

            condition_vectors[:,:,:,:] = 0

            color_idx = hair_color_list.index(color)
            condition_vectors[:,color_idx,:,:] = 100
            ####################################################################

            ############################################################################

            # ground truth color info
            print 'color =', color
            print 'color idx =', hair_color_list.index(color)
            ground_truth_class = np.zeros((BATCH_SIZE, 9), dtype=float)
            ground_truth_class[:, hair_color_list.index(color)] = 1
            #print 'ground_truth_class =', ground_truth_class
            
            if is_gpu_mode:
                inputs = Variable(torch.from_numpy(input_img).float().cuda())
                answers = Variable(torch.from_numpy(answer_img).float().cuda())
                ground_truth_class = Variable(torch.from_numpy(ground_truth_class).float().cuda())
            else:
                inputs = Variable(torch.from_numpy(input_img).float())
                answers = Variable(torch.from_numpy(answer_img).float())
                ground_truth_class = Variable(torch.from_numpy(ground_truth_class).float())

            # feedforward the inputs. generators.
            outputs_gen_a_to_b = gen_model_a(torch.cat((condition_vectors, inputs), 1))
            outputs_gen_b_to_a = gen_model_b(torch.cat((condition_vectors, answers), 1))

            # feedforward the data to the discriminator_a
            #output_disc_real_a, class_real_a = disc_model_a(torch.cat((condition_vectors, inputs), 1))
            output_disc_real_a, class_real_a = disc_model_a(inputs)
            
            #output_disc_fake_a, class_fake_a = disc_model_a(torch.cat((condition_vectors, outputs_gen_b_to_a), 1))
            output_disc_fake_a, class_fake_a = disc_model_a(outputs_gen_b_to_a)

            # feedforward the data to the discriminator_b
            #output_disc_real_b, class_real_b = disc_model_b(torch.cat((condition_vectors, answers), 1))
            output_disc_real_b, class_real_b = disc_model_b(answers)
            print 'output_disc_real_b =', output_disc_real_b
            print 'output_class_real_b =', class_real_b
            
            #output_disc_fake_b, class_fake_b = disc_model_b(torch.cat((condition_vectors, outputs_gen_a_to_b), 1))
            output_disc_fake_b, class_fake_b = disc_model_b(outputs_gen_a_to_b)
            print 'output_disc_fake_b =', output_disc_fake_b
            print 'output_class_fake_b =', class_fake_b

            # loss functions
            # lsgan loss for the discriminator_a
            loss_disc_a_lsgan = 0.5 * (torch.mean((output_disc_real_a - 1) ** 2) + torch.mean(output_disc_fake_a ** 2))

            # lsgan loss for the discriminator_b
            loss_disc_b_lsgan = 0.5 * (torch.mean((output_disc_real_b - 1) ** 2) + torch.mean(output_disc_fake_b ** 2))
            
            # class loss for the discriminator_a (not necessary. since only black color exists)
            
            # class loss for the discriminator_b
            ground_truth_class_view = ground_truth_class.view(BATCH_SIZE * 9)
            class_real_b_view = class_real_b.view(BATCH_SIZE * 9)
            class_fake_b_view = class_fake_b.view(BATCH_SIZE * 9)
            loss_disc_b_class = criterion_cross_ent(class_real_b_view, ground_truth_class_view) +\
                                criterion_cross_ent(class_fake_b_view, ground_truth_class_view)
            
            # total loss (disc)
            #total_disc_loss_a = loss_disc_a_lsgan + loss_disc_a_class
            
            total_disc_loss_b = loss_disc_b_lsgan + loss_disc_b_class
            print 'loss comparison'
            print 'loss_disc_b_class =', loss_disc_b_class
            print 'loss_disc_b_lsgan =', loss_disc_b_lsgan
            
            # cycle-consistency loss(a)
            reconstructed_a = gen_model_b(torch.cat((condition_vectors, outputs_gen_a_to_b), 1))
            l1_loss_rec_a = F.l1_loss(reconstructed_a, inputs)

            # cycle-consistency loss(b)
            reconstructed_b = gen_model_a(torch.cat((condition_vectors, outputs_gen_b_to_a), 1))
            l1_loss_rec_b = F.l1_loss(reconstructed_b, answers)

            # lsgan loss for the generator_a
            loss_gen_lsgan_a = 0.5 * torch.mean((output_disc_fake_b - 1) ** 2)

            # lsgan loss for the generator_b
            loss_gen_lsgan_b = 0.5 * torch.mean((output_disc_fake_a - 1) ** 2)

            loss_gen_total_lsgan = loss_gen_lsgan_a + loss_gen_lsgan_b + 0.005 * (l1_loss_rec_a + l1_loss_rec_b)

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
            total_disc_loss_b.backward(retain_graph=True)
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
                logger.scalar_summary(color + ':loss(generator_a)', loss_gen_lsgan_a, i)
                logger.scalar_summary(color + ':loss(generator_b)', loss_gen_lsgan_b, i)
                logger.scalar_summary(color + ':loss-rec_a-l1', l1_loss_rec_a, i)
                logger.scalar_summary(color + ':loss-rec_b-l1', l1_loss_rec_b, i)
                logger.scalar_summary(color + ':loss(discriminator_a)', loss_disc_a_lsgan, i)
                logger.scalar_summary(color + ':loss(discriminator_b)', loss_disc_b_lsgan, i)
                #logger.scalar_summary(color + ':disc_a-out-for-real', output_disc_real_a[0], i)
                #logger.scalar_summary(color + ':disc_a-out-for-fake', output_disc_fake_a[0], i)
                #logger.scalar_summary(color + ':disc_b-out-for-real', output_disc_real_b[0], i)
                #logger.scalar_summary(color + ':disc_b-out-for-fake', output_disc_fake_b[0], i)

                # tf-board (images - first 1 batches)
                inputs_imgs_temp = inputs.cpu().data.numpy()[0:BATCH_SIZE]
                output_imgs_temp_b = outputs_gen_a_to_b.cpu().data.numpy()[0:BATCH_SIZE]
                output_imgs_temp_a = outputs_gen_b_to_a.cpu().data.numpy()[0:BATCH_SIZE]
                answer_imgs_temp = answers.cpu().data.numpy()[0:BATCH_SIZE]
                reconstructed_a_temp = reconstructed_a.cpu().data.numpy()[0:BATCH_SIZE]
                reconstructed_b_temp = reconstructed_b.cpu().data.numpy()[0:BATCH_SIZE]

                # logger.an_image_summary('generated', output_img, i)
                logger.image_summary(color + ':input', inputs_imgs_temp, i)
                logger.image_summary(color + ':generated_b', output_imgs_temp_b, i)
                logger.image_summary(color + ':generated_a', output_imgs_temp_a, i)
                logger.image_summary(color + ':reconstructed_a', reconstructed_a_temp, i)
                logger.image_summary(color + ':reconstructed_b', reconstructed_b_temp, i)
                logger.image_summary(color + ':real', answer_imgs_temp, i)
            
        # prepare for the next iter
        saved_image_buff_read_index = image_buff_read_index
       
        # iterate through colors and manipulate the input data and condition vectors
        ############################################################################
        
        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(gen_model_a.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'acgan_unified_cycle_gen_model_iter_' + str(i) + '.pt')
