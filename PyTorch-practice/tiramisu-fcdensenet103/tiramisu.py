import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sub_modules import Layer, TransitionUp, TransitionDown

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

        model = [Layer(32)]

        #model += [TransitionUp(32)]

        #model += [TransitionDown(32)]

        self.model = nn.Sequential(*model)

        '''
        # input size is 32x32x3 for cifar-10 dataset
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, bias=True)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, bias=True)

        # size of the feature maps would be 4x4 at this point
        self.fc1 = nn.Linear(6400, 500)
        self.fc2 = nn.Linear(500, 10)
        '''

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 6400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.log_softmax(x)
        '''

        return self.model(x)

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



