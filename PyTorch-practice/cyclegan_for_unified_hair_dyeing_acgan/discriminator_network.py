import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader_for_unified_cyclegan import hair_color_list

BATCH_SIZE = 1

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
        #self.first_conv_layer = ConvolutionDown(in_channels=3+len(hair_color_list), out_channels=32, kernel_size=3)
        self.first_conv_layer = ConvolutionDown(in_channels=3, out_channels=128, kernel_size=3)
        self.second_conv_layer = ConvolutionDown(in_channels=128, out_channels=256, kernel_size=3)
        self.third_conv_layer = ConvolutionDown(in_channels=256, out_channels=512, kernel_size=3)
        self.fourth_conv_layer = ConvolutionDown(in_channels=512, out_channels=1024, kernel_size=3)
        self.fifth_conv_layer = ConvolutionDown(in_channels=1024, out_channels=1024, kernel_size=3)
        self.last_conv_layer = ConvolutionDown(in_channels=1024, out_channels=10, kernel_size=3)
        
        # auxiliary classifier (for 9 colors)
        self.fc_aux = nn.Linear(4 * 4 * 10, 9)
        
        # real/fake
        self.fc_disc = nn.Linear(4 * 4 * 10, 1)

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
               
        # auxiliary classifier branch (for 9 colors)
        x = x.view(BATCH_SIZE, 4 * 4 * 10)
        x_aux = F.relu(self.fc_aux(x))
        class_out = nn.functional.softmax(x_aux)
        
        #print 'class_out =', class_out
        
        x_disc = F.relu(self.fc_disc(x))
        disc_out = nn.functional.sigmoid(x_disc)
        
        #print 'disc_out =', disc_out

        return disc_out, class_out
