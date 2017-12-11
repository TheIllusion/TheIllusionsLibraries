import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sub_modules import Layer, TransitionUp, TransitionDown, DenseBlock
import time, cv2
import numpy as np
import data_loader

# gpu mode
is_gpu_mode = False

# batch size
BATCH_SIZE = 1
TOTAL_ITERATION = 10000

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 1000
# i7-2600k
#MODEL_SAVING_DIRECTORY = '/home/illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/tiramisu-fcdensenet103/models/'
# macbook pro
MODEL_SAVING_DIRECTORY = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/tiramisu-fcdensenet103/models/'

# macbook pro
#cifar10_data_dir = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/fashion-mnist/data/'

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# cifar10 dataset
'''
CIFA-10 dataset (Docs: https://www.cs.toronto.edu/~kriz/cifar.html)
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
'''

'''
train_set = datasets.CIFAR10(cifar10_data_dir, train=True,
                             transform=transform, target_transform=None,
                             download=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_set = datasets.CIFAR10(cifar10_data_dir, train=False,
                            transform=transform, target_transform=None,
                            download=True)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
'''

class Tiramisu(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Tiramisu, self).__init__()

        # define parameters
        # first convolution
        self.first_conv_layer = Layer(kernel_size=3, in_channels=3, out_channels=48)

        # first dense block
        self.first_dense_block = DenseBlock(layers=4, in_channels=48, k_feature_maps=16, is_gpu_mode=is_gpu_mode)
        # first transition down
        self.first_transition_down = TransitionDown(112)

        # second dense block
        self.second_dense_block = DenseBlock(layers=5, in_channels=112, k_feature_maps=16, is_gpu_mode=is_gpu_mode)
        # second transition down
        self.second_transition_down = TransitionDown(192)
        # third dense block

        self.third_dense_block = DenseBlock(layers=7, in_channels=192, k_feature_maps=16, is_gpu_mode=is_gpu_mode)
        # third transition down
        self.third_transition_down = TransitionDown(304)

        # fourth dense block
        self.fourth_dense_block = DenseBlock(layers=10, in_channels=304, k_feature_maps=16, is_gpu_mode=is_gpu_mode)
        # fourth transition down
        self.fourth_transition_down = TransitionDown(464)

        # fifth dense block
        self.fifth_dense_block = DenseBlock(layers=12, in_channels=464, k_feature_maps=16, is_gpu_mode=is_gpu_mode)
        # fifth transition down
        self.fifth_transition_down = TransitionDown(656)

        # middle dense block
        self.middle_dense_block = DenseBlock(layers=15, in_channels=656, k_feature_maps=16, is_gpu_mode=is_gpu_mode)

        # later-first transition up
        #self.later_first_transition_up = TransitionUp(896)
        self.later_first_transition_up = TransitionUp(240)
        # later-first dense block
        self.later_first_dense_block = DenseBlock(layers=12, in_channels=896, k_feature_maps=16, is_gpu_mode=is_gpu_mode)

        # later-second transition up
        #self.later_second_transition_up = TransitionUp(1088)
        self.later_second_transition_up = TransitionUp(192)
        # later-second dense block
        self.later_second_dense_block = DenseBlock(layers=10, in_channels=656, k_feature_maps=16, is_gpu_mode=is_gpu_mode)

        # later-third transition up
        self.later_third_transition_up = TransitionUp(160)
        # later-third dense block
        self.later_third_dense_block = DenseBlock(layers=7, in_channels=464, k_feature_maps=16, is_gpu_mode=is_gpu_mode)

        # later-fourth transition up
        self.later_fourth_transition_up = TransitionUp(112)
        # later-fourth dense block
        self.later_fourth_dense_block = DenseBlock(layers=5, in_channels=304, k_feature_maps=16, is_gpu_mode=is_gpu_mode)

        # later-fifth transition up
        self.later_fifth_transition_up = TransitionUp(80)
        # later-fifth dense block
        self.later_fifth_dense_block = DenseBlock(layers=4, in_channels=192, k_feature_maps=16, is_gpu_mode=is_gpu_mode)

        # last convolution - cifar10 has 10 classes
        #self.last_conv_layer = Layer(64, 10)

        # hair dataset (3 classes)
        self.last_conv_layer = Layer(kernel_size=3, in_channels=64, out_channels=3)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        ######################################################################
        # define the forward connections
        ######################################################################
        x_first_conv_out = self.first_conv_layer(x)

        x_first_dense_out = self.first_dense_block(x_first_conv_out)
        # concatenate filters
        x_first_dense_out_concat = torch.cat((x_first_conv_out, x_first_dense_out), 1)
        x_first_td_out = self.first_transition_down(x_first_dense_out_concat)

        x_second_dense_out = self.second_dense_block(x_first_td_out)
        # concatenate filters
        x_second_dense_out_concat = torch.cat((x_first_td_out, x_second_dense_out), 1)
        x_second_td_out = self.second_transition_down(x_second_dense_out_concat)

        x_third_dense_out = self.third_dense_block(x_second_td_out)
        # concatenate filters
        x_third_dense_out_concat = torch.cat((x_second_td_out, x_third_dense_out), 1)
        x_third_td_out = self.third_transition_down(x_third_dense_out_concat)

        x_fourth_dense_out = self.fourth_dense_block(x_third_td_out)
        # concatenate filters
        x_fourth_dense_out_concat = torch.cat((x_third_td_out, x_fourth_dense_out), 1)
        x_fourth_td_out = self.fourth_transition_down(x_fourth_dense_out_concat)

        x_fifth_dense_out = self.fifth_dense_block(x_fourth_td_out)
        # concatenate filters
        x_fifth_dense_out_concat = torch.cat((x_fourth_td_out, x_fifth_dense_out), 1)
        x_fifth_td_out = self.fifth_transition_down(x_fifth_dense_out_concat)

        x_middle_dense_out = self.middle_dense_block(x_fifth_td_out)

        ######################################################################
        # define the backward connections
        ######################################################################
        x_later_first_tu_out = self.later_first_transition_up(x_middle_dense_out)
        # concatenate filters (Skip-Connection)
        x_later_first_tu_out_concat = torch.cat((x_later_first_tu_out, x_fifth_dense_out_concat), 1)
        x_later_first_dense_out = self.later_first_dense_block(x_later_first_tu_out_concat)

        x_later_second_tu_out = self.later_second_transition_up(x_later_first_dense_out)
        # concatenate filters (Skip-Connection)
        x_later_second_tu_out_concat = torch.cat((x_later_second_tu_out, x_fourth_dense_out_concat), 1)
        x_later_second_dense_out = self.later_second_dense_block(x_later_second_tu_out_concat)

        x_later_third_tu_out = self.later_third_transition_up(x_later_second_dense_out)
        # concatenate filters (Skip-Connection)
        x_later_third_tu_out_concat = torch.cat((x_later_third_tu_out, x_third_dense_out_concat), 1)
        x_later_third_dense_out = self.later_third_dense_block(x_later_third_tu_out_concat)

        x_later_fourth_tu_out = self.later_fourth_transition_up(x_later_third_dense_out)
        # concatenate filters (Skip-Connection)
        x_later_fourth_tu_out_concat = torch.cat((x_later_fourth_tu_out, x_second_dense_out_concat), 1)
        x_later_fourth_dense_out = self.later_fourth_dense_block(x_later_fourth_tu_out_concat)

        x_later_fifth_tu_out = self.later_fifth_transition_up(x_later_fourth_dense_out)
        # concatenate filters (Skip-Connection)
        x_later_fifth_tu_out_concat = torch.cat((x_later_fifth_tu_out, x_first_dense_out_concat), 1)
        x_later_fifth_dense_out = self.later_fifth_dense_block(x_later_fifth_tu_out_concat)

        x_last_conv_out = self.last_conv_layer(x_later_fifth_dense_out)

        sigmoid_out = nn.functional.sigmoid(x_last_conv_out)
        #softmax_out = nn.functional.softmax(x_last_conv_out)

        output = sigmoid_out * 255

        return output

if __name__ == "__main__":
    print 'main'

    tiramisu_model = Tiramisu()

    if is_gpu_mode:
        tiramisu_model.cuda()

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Variables it should update.
    learning_rate = 1e-5

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(size_average=False)
    #criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(tiramisu_model.parameters(), lr=learning_rate)

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

    '''            
    for epoch in range(50):
        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data

            # wrap them in Variables
            if is_gpu_mode:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
    '''

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

            data_loader.buff_status[image_buff_read_index] = 'empty'

            image_buff_read_index = image_buff_read_index + 1
            if image_buff_read_index >= data_loader.image_buffer_size:
                image_buff_read_index = 0

            if is_gpu_mode:
                inputs, answers = Variable(torch.from_numpy(input_img).float().cuda()), \
                                  Variable(torch.from_numpy(answer_img).float().cuda())
            else:
                inputs, answers = torch.from_numpy(input_img).float(), torch.from_numpy(answer_img).float()

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        outputs = tiramisu_model(inputs)

        # flatten data
        outputs_view = outputs.view(-1, 3 * data_loader.INPUT_IMAGE_WIDTH * data_loader.INPUT_IMAGE_HEIGHT)
        answers_view = answers.view(-1, 3 * data_loader.INPUT_IMAGE_WIDTH * data_loader.INPUT_IMAGE_HEIGHT)

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
            torch.save(tiramisu_model.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'tiramisu_iter_' +str(i) + '.pt')

    _, predicted = torch.max(outputs.data, 1)
    print 'output = ', predicted
    #print 'target cls = ', labels
