import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sub_modules import Layer, TransitionUp, TransitionDown
import time, cv2
import numpy as np

# gpu mode
is_gpu_mode = True 

# feedforward mode
if __name__ == '__main__':
    is_feedforward_mode = False
else:
    is_feedforward_mode = True

# learning rate
LEARNING_RATE = 1 * 1e-4

if not is_feedforward_mode:
    import data_loader
    
# batch size
BATCH_SIZE = 5
TOTAL_ITERATION = 2000000

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 10000

# zero centered
MEAN_VALUE_FOR_ZERO_CENTERED = 128 

# tbt005
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/unet/models/'

class Unet(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Unet, self).__init__()

        # unet architecture
        
        # first conv
        self.first_conv_layer = Layer(kernel_size=3, in_channels=3, out_channels=64)
        
        # second conv
        self.second_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=64)
        
        # max pool
        self.first_transition_down = TransitionDown(64)
        
        # third conv
        self.third_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=128)
        
        # fourth conv
        self.fourth_conv_layer = Layer(kernel_size=3, in_channels=128, out_channels=128)
        
        # max pool
        self.second_transition_down = TransitionDown(128)
        
        # fifth conv
        self.fifth_conv_layer = Layer(kernel_size=3, in_channels=128, out_channels=256)
        
        # sixth conv
        self.sixth_conv_layer = Layer(kernel_size=3, in_channels=256, out_channels=256)
        
        # max pool
        self.third_transition_down = TransitionDown(256)
        
        # seventh conv
        self.seventh_conv_layer = Layer(kernel_size=3, in_channels=256, out_channels=512)
        
        # eighth conv
        self.eighth_conv_layer = Layer(kernel_size=3, in_channels=512, out_channels=512)
        
        # max pool
        self.fourth_transition_down = TransitionDown(512)
        
        # ninth conv
        self.ninth_conv_layer = Layer(kernel_size=3, in_channels=512, out_channels=1024)
        
        # tenth conv
        self.tenth_conv_layer = Layer(kernel_size=3, in_channels=1024, out_channels=1024)
        
        # first Transition Up
        self.later_first_transition_up = TransitionUp(1024)
        
        # followed by 2 conv layers
        self.later_first_conv_layer = Layer(kernel_size=3, in_channels=1024, out_channels=512)
        self.later_second_conv_layer = Layer(kernel_size=3, in_channels=512, out_channels=512)
        
        # second Transition Up
        self.later_second_transition_up = TransitionUp(512)
        
        # followed by 2 conv layers 
        self.later_third_conv_layer = Layer(kernel_size=3, in_channels=512, out_channels=256)
        self.later_fourth_conv_layer = Layer(kernel_size=3, in_channels=256, out_channels=256)
        
        # third Transition Up
        self.later_third_transition_up = TransitionUp(256)
        
        # followed by 2 conv layers 
        self.later_fifth_conv_layer = Layer(kernel_size=3, in_channels=256, out_channels=128)
        self.later_sixth_conv_layer = Layer(kernel_size=3, in_channels=128, out_channels=128)
        
        # fourth Transition Up
        self.later_fourth_transition_up = TransitionUp(128)
        
        # followed by 2 conv layers 
        self.later_seventh_conv_layer = Layer(kernel_size=3, in_channels=128, out_channels=64)
        self.later_eighth_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=64)
        
        # last conv layer (num. of output channels should be modified according to the number of classes)
        self.later_last_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=3)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        # unet
        ######################################################################
        # define the forward connections
        ######################################################################
        x = self.first_conv_layer(x)
        x_second_conv_out = self.second_conv_layer(x)
        
        # first TD
        x = self.first_transition_down(x_second_conv_out)
        
        x = self.third_conv_layer(x)
        x_fourth_conv_out = self.fourth_conv_layer(x)
        
        # second TD
        x = self.second_transition_down(x_fourth_conv_out)
        
        x = self.fifth_conv_layer(x)
        x_sixth_conv_out = self.sixth_conv_layer(x)
        
        # third TD
        x = self.third_transition_down(x_sixth_conv_out)
        
        x = self.seventh_conv_layer(x)
        x_eighth_conv_out = self.eighth_conv_layer(x)
        
        # fourth TD
        x = self.fourth_transition_down(x_eighth_conv_out)
        
        ######################################################################
        # define the middle connections
        ######################################################################
        x = self.ninth_conv_layer(x)
        x = self.tenth_conv_layer(x)
        
        ######################################################################
        # define the backward connections
        ######################################################################
        # first TU
        x = self.later_first_transition_up(x)
        x_later_first_concat = torch.cat((x, x_eighth_conv_out), 1)
        
        # followed by 2 conv layers
        x = self.later_first_conv_layer(x_later_first_concat)
        x = self.later_second_conv_layer(x)
        
        # second TU
        x = self.later_second_transition_up(x)
        x_later_second_concat = torch.cat((x, x_sixth_conv_out), 1)
        
        # followed by 2 conv layers
        x = self.later_third_conv_layer(x_later_second_concat)
        x = self.later_fourth_conv_layer(x)
        
        # third TU
        x = self.later_third_transition_up(x)
        x_later_third_concat = torch.cat((x, x_fourth_conv_out), 1)
        
        # followed by 2 conv layers
        x = self.later_fifth_conv_layer(x_later_third_concat)
        x = self.later_sixth_conv_layer(x)
        
        # fourth TU
        x = self.later_fourth_transition_up(x)
        x_later_fourth_concat = torch.cat((x, x_second_conv_out), 1)
        
        # followed by 2 conv layers
        x = self.later_seventh_conv_layer(x_later_fourth_concat)
        x = self.later_eighth_conv_layer(x)
        
        # last conv layer
        x = self.later_last_conv_layer(x)

        sigmoid_out = nn.functional.sigmoid(x)
        #softmax_out = custom_softmax_for_segmentation(x_last_conv_out, 3)

        #output = sigmoid_out * 255
        #output = softmax_out

        return sigmoid_out

if __name__ == "__main__":
    print 'main'

    unet_model = Unet()

    if is_gpu_mode:
        unet_model.cuda()
        #unet_model = torch.nn.DataParallel(unet_model).cuda()

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Variables it should update.
    learning_rate = LEARNING_RATE

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    #criterion = torch.nn.MSELoss(size_average=False)
    criterion = torch.nn.BCELoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)

    # read imgs
    image_buff_read_index = 0

    # opencv style
    '''
    input_img = np.empty(shape=(BATCH_SIZE, 512, 512, 3))
    answer_img = np.empty(shape=(BATCH_SIZE, 512, 512, 3))
    '''

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    answer_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))

    for i in range(TOTAL_ITERATION):

        # print 'iter: ', str(iter)
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

            # make zero-centered (this could be different from the original unet)
            input_img[j] = input_img[j] - MEAN_VALUE_FOR_ZERO_CENTERED
            answer_img[j] = answer_img[j] - MEAN_VALUE_FOR_ZERO_CENTERED
            
            data_loader.buff_status[image_buff_read_index] = 'empty'

            image_buff_read_index = image_buff_read_index + 1
            if image_buff_read_index >= data_loader.image_buffer_size:
                image_buff_read_index = 0

            if is_gpu_mode:
                inputs, answers = Variable(torch.from_numpy(input_img).float().cuda()), \
                                  Variable(torch.from_numpy(answer_img).float().cuda())
            else:
                inputs, answers = Variable(torch.from_numpy(input_img).float()), Variable(torch.from_numpy(answer_img).float())

        answers = answers / 255.
        #answers = answers.long()

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        outputs = unet_model(inputs)

        # change the dimension
        outputs_view = outputs.view(BATCH_SIZE * data_loader.INPUT_IMAGE_WIDTH * data_loader.INPUT_IMAGE_HEIGHT * 3)
        answers_view = answers.view(BATCH_SIZE * data_loader.INPUT_IMAGE_WIDTH * data_loader.INPUT_IMAGE_HEIGHT * 3)

        #loss = criterion(outputs, answers)
        loss = criterion(outputs_view, answers_view)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        if i % 10 == 0:
            print '-----------------------------------'
            print 'iterations = ', str(i)
            print 'loss = ', str(loss)

        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(unet_model.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'unet_lr_0_0001_total_augmented_iter_' +str(i) + '.pt')
