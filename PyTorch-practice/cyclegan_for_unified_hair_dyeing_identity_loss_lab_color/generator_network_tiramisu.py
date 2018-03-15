import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from data_loader_for_unified_cyclegan import hair_color_list

# Layer module from the paper - Table 1.
class Layer(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(Layer, self).__init__()

        self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=1, bias=True)

        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)

        return x

# Transition Down (TD) module from the paper - Table 1.
class TransitionDown(nn.Module):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()

        self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=1, stride=1, bias=True)

        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)

        out_channels = in_channels
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.max_pool2d(input=x, kernel_size=2)
        x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)

        return x


# Transition Up (TU) module from the paper - Table 1.
class TransitionUp(nn.Module):
    def __init__(self, in_channels):
        super(TransitionUp, self).__init__()

        '''
        # nn.ConvTranspose2d
        Args
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution
        padding (int or tuple, optional): Zero-padding added to both sides of the input
        output_padding (int or tuple, optional): Zero-padding added to one side of the output
        groups (int, optional): Number of blocked connections from input channels to output channels
        bias (bool, optional): If True, adds a learnable bias to the output
        dilation (int or tuple, optional): Spacing between kernel elements
        '''

        self.transpoed_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                                 kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

        # weight initialization
        torch.nn.init.xavier_uniform(self.transpoed_conv.weight)

    def forward(self, x):
        x = self.transpoed_conv(x)
        return x


# Dense Block - Figure 2.
class DenseBlock(nn.Module):
    def __init__(self, layers, in_channels, k_feature_maps):
        super(DenseBlock, self).__init__()

        self.num_layers = layers
        self.layers_list = []

        # add layers to the list
        for i in xrange(layers):
            self.layers_list.append(Layer(kernel_size=3,
                                          in_channels=in_channels + i * k_feature_maps,
                                          out_channels=k_feature_maps))

            # add_module to this instance
            self.add_module('denseblock_idx_' + str(i), self.layers_list[i])

    def forward(self, x):

        # feedforward x to the first layer and add the result to the list
        x_first_out = self.layers_list[0].forward(x)

        # initialize the list
        forwarded_output_list = []

        forwarded_output_list.append(x_first_out)
        prev_x = x

        # feedforward process from the second to the last layer
        for i in range(1, self.num_layers):
            # concatenate filters
            concatenated_filters = torch.cat((forwarded_output_list[i - 1], prev_x), 1)
            # forward
            x_next_out = self.layers_list[i].forward(concatenated_filters)
            # add to the list
            forwarded_output_list.append(x_next_out)
            # prepare the temporary variable for the next loop
            prev_x = concatenated_filters

        # prepare the output (this will have (k_feature_maps * layers) feature maps)
        output_x = torch.cat(forwarded_output_list, 1)

        return output_x

class Tiramisu(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Tiramisu, self).__init__()

        # define parameters
        # first convolution
        self.first_conv_layer = Layer(kernel_size=3, in_channels=3+len(hair_color_list), out_channels=48)

        # first dense block
        self.first_dense_block = DenseBlock(layers=4, in_channels=48, k_feature_maps=16)
        # first transition down
        self.first_transition_down = TransitionDown(112)

        # second dense block
        self.second_dense_block = DenseBlock(layers=5, in_channels=112, k_feature_maps=16)
        # second transition down
        self.second_transition_down = TransitionDown(192)
        # third dense block

        self.third_dense_block = DenseBlock(layers=7, in_channels=192, k_feature_maps=16)
        # third transition down
        self.third_transition_down = TransitionDown(304)

        # fourth dense block
        self.fourth_dense_block = DenseBlock(layers=10, in_channels=304, k_feature_maps=16)
        # fourth transition down
        self.fourth_transition_down = TransitionDown(464)

        # fifth dense block
        self.fifth_dense_block = DenseBlock(layers=12, in_channels=464, k_feature_maps=16)
        # fifth transition down
        self.fifth_transition_down = TransitionDown(656)

        # middle dense block
        self.middle_dense_block = DenseBlock(layers=15, in_channels=656, k_feature_maps=16)

        # later-first transition up
        #self.later_first_transition_up = TransitionUp(896)
        self.later_first_transition_up = TransitionUp(240)
        # later-first dense block
        self.later_first_dense_block = DenseBlock(layers=12, in_channels=896, k_feature_maps=16)

        # later-second transition up
        #self.later_second_transition_up = TransitionUp(1088)
        self.later_second_transition_up = TransitionUp(192)
        # later-second dense block
        self.later_second_dense_block = DenseBlock(layers=10, in_channels=656, k_feature_maps=16)

        # later-third transition up
        self.later_third_transition_up = TransitionUp(160)
        # later-third dense block
        self.later_third_dense_block = DenseBlock(layers=7, in_channels=464, k_feature_maps=16)

        # later-fourth transition up
        self.later_fourth_transition_up = TransitionUp(112)
        # later-fourth dense block
        self.later_fourth_dense_block = DenseBlock(layers=5, in_channels=304, k_feature_maps=16)

        # later-fifth transition up
        self.later_fifth_transition_up = TransitionUp(80)
        # later-fifth dense block
        self.later_fifth_dense_block = DenseBlock(layers=4, in_channels=192, k_feature_maps=16)

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
        #softmax_out = custom_softmax_for_segmentation(x_last_conv_out, 3)

        output = sigmoid_out * 255
        #output = softmax_out

        return output