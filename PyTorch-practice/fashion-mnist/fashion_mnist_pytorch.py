import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# macbook pro
root_dir = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/PyTorch-practice/fashion-mnist/data/'

class CNNBasic(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(CNNBasic, self).__init__()

        # input size is 32x32x3 for cifar-10 dataset
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3, stride=2, bias=True)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=2, bias=True)

        # size of the feature maps would be 4x4 at this point
        self.fc1 = nn.Linear(980, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 980)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.log_softmax(x)

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

'''
CIFA-10 dataset (Docs: https://www.cs.toronto.edu/~kriz/cifar.html)
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
'''
train_loader = DataLoader(datasets.CIFAR10(root_dir, train=True, transform=transform, target_transform=None, download=True))
test_loader = DataLoader(datasets.CIFAR10(root_dir, train=False, transform=transform, target_transform=None, download=True))

if __name__ == "__main__":

    # create a cnn model
    cnn_basic_model = CNNBasic()

    loss_fn = torch.nn.MSELoss(size_average=False)

    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Variables it should update.
    learning_rate = 1e-4

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    #criterion = torch.nn.MSELoss(size_average=False)
    criterion = nn.CrossEntropyLoss()

    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(cnn_basic_model.parameters(), lr=learning_rate)

    # load data from cifar10
    idx = 0
    for batch_idx, (data, target) in enumerate(train_loader):
    #for data, target in train_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        data, target = Variable(data), Variable(target)

        output = cnn_basic_model(data)
        loss = criterion(output, target)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        if idx % 10 == 0:
            print 'idx = ', str(idx)
            print 'loss = ', str(loss)

        idx = idx + 1

    print 'hi'
