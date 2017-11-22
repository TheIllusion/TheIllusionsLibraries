import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sub_modules import Layer, TransitionUp, TransitionDown, DenseBlock

# gpu mode
is_gpu_mode = False

# macbook pro
cifar10_data_dir = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/fashion-mnist/data/'

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# cifar10 dataset
train_set = datasets.CIFAR10(cifar10_data_dir, train=True,
                             transform=transform, target_transform=None,
                             download=True)

train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=2)

test_set = datasets.CIFAR10(cifar10_data_dir, train=False,
                            transform=transform, target_transform=None,
                            download=True)

test_loader = DataLoader(test_set, batch_size=30, shuffle=False, num_workers=2)

class Tiramisu(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Tiramisu, self).__init__()

        # define parameters
        # first convolution
        self.first_conv_layer = Layer(48)
        # first dense block
        self.first_dense_block = DenseBlock(112)
        # first transition down
        self.first_transition_down = TransitionDown(32)
        # second transition down
        self.second_transition_down = TransitionDown(32)
        # middle dense block
        self.middle_dense_block = DenseBlock(32)
        # later-first transition up
        self.later_first_transition_up = TransitionUp(32)
        # later-first dense block
        self.later_first_dense_block = DenseBlock(32)
        # later-second transition up
        self.later_second_transition_up = TransitionUp(32)
        # later-second dense block
        self.later_second_dense_block = DenseBlock(32)
        # last convolution
        self.last_conv_layer = Layer(32)

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
        x_first_dense_out_concat = torch.cat((x_first_conv_out, x_first_dense_out), 0)
        x_first_td_out = self.first_transition_down(x_first_dense_out_concat)
        x_second_dense_out = self.later_second_dense_block(x_first_td_out)
        # concatenate filters
        x_second_dense_out_concat = torch.cat((x_first_td_out, x_second_dense_out), 0)
        x_second_td_out = self.second_transition_down(x_second_dense_out_concat)
        x_middle_dense_out = self.middle_dense_block(x_second_td_out)

        ######################################################################
        # define the backward connections
        ######################################################################
        x_later_first_tu_out = self.later_first_transition_up(x_middle_dense_out)
        # concatenate filters (Skip-Connection)
        x_later_first_tu_out_concat = torch.cat((x_later_first_tu_out, x_second_dense_out_concat), 0)
        x_later_first_dense_out = self.later_first_dense_block(x_later_first_tu_out_concat)
        x_later_second_tu_out = self.later_second_transition_up(x_later_first_dense_out)
        # concatenate filters (Skip-Connection)
        x_later_second_tu_out_concat = torch.cat((x_later_second_tu_out, x_first_dense_out_concat), 0)
        x_later_second_dense_out = self.later_second_dense_block(x_later_second_tu_out_concat)
        x_last_conv_out = self.last_conv_layer(x_later_second_dense_out)
        return x_last_conv_out

if __name__ == "__main__":
    print 'main'

    tiramisu_model = Tiramisu()

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Variables it should update.
    learning_rate = 1e-5

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(size_average=False)
    # criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(tiramisu_model.parameters(), lr=learning_rate)

    for epoch in range(50):
        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data

            # wrap them in Variables
            if is_gpu_mode:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            outputs = tiramisu_model(inputs)
            #loss = criterion(outputs, labels)
            loss = criterion(outputs, inputs)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()

            if i % 100 == 0:
                print '-----------------------------------'
                print 'i = ', str(i)
                print 'loss = ', str(loss)

        _, predicted = torch.max(outputs.data, 1)
        print 'output = ', predicted
        print 'target cls = ', labels



