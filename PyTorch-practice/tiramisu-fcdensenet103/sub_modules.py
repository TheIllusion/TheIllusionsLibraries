import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Layer module from the paper
class Layer(nn.Module):
    def __init__(self):
        print 'hi'
    def forward(self, x):
        print 'forward'

# Transition Down (TD) module from the paper
class TransitionDown(nn.Module):
    def __init__(self):
        print 'hi'
    def forward(self, x):
        print 'forward'

# Transition Up (TU) module from the paper
class TransitionUp(nn.Module):
    def __init__(self):
        print 'hi'

    def forward(self, x):
        print 'forward'