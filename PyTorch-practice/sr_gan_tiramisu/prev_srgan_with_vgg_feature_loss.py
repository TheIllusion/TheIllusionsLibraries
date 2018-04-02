import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2, time, os
import numpy as np
from torchvision import models, transforms
import data_loader_for_pix2pix_with_pil as data_loader
from logger import Logger
from generator_modified_tiramisu import Tiramisu

print 'srgan_with_vgg_feature_loss.py'

# gpu mode
is_gpu_mode = torch.cuda.is_available()

# batch size
BATCH_SIZE = 1
TOTAL_ITERATION = 1000000

# learning rate
LEARNING_RATE_GENERATOR = 3 * 1e-4
LEARNING_RATE_DISCRIMINATOR = 1 * 1e-4

# zero centered
# MEAN_VALUE_FOR_ZERO_CENTERED = 128

# model saving (iterations)
MODEL_SAVING_FREQUENCY = 5000

# T005
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/sr_gan_tiramisu/generator_checkpoints_with_vgg_loss/'

TF_BOARD_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/sr_gan_tiramisu/tfboard_vgg_loss/'

if not os.path.exists(MODEL_SAVING_DIRECTORY):
    os.mkdir(MODEL_SAVING_DIRECTORY)

if not os.path.exists(TF_BOARD_DIRECTORY):
    os.mkdir(TF_BOARD_DIRECTORY)

# tensor-board logger
logger = Logger(TF_BOARD_DIRECTORY)

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TransitionDown, self).__init__()

        # self.drop_out = nn.Dropout2d(p=0.2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, padding=1, stride=2, bias=False)

        # weight initialization
        torch.nn.init.xavier_uniform(self.conv.weight)

        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x = F.max_pool2d(input=x, kernel_size=2)
        # x = self.drop_out(x)
        x = F.relu(self.conv(x))
        x = self.batch_norm(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate our custom modules and assign them as
        member variables.
        """
        super(Discriminator, self).__init__()

        # input image will have the size of 1024x1024x3
        self.first_conv_layer = TransitionDown(in_channels=3, out_channels=64, kernel_size=3)
        # 512x512
        self.second_conv_layer = TransitionDown(in_channels=64, out_channels=128, kernel_size=3)
        # 256x256
        self.third_conv_layer = TransitionDown(in_channels=128, out_channels=256, kernel_size=3)
        # 128x128
        self.fourth_conv_layer = TransitionDown(in_channels=256, out_channels=512, kernel_size=3)
        # 64x64
        self.fifth_conv_layer = TransitionDown(in_channels=512, out_channels=512, kernel_size=3)
        # 32x32
        self.last_conv_layer = TransitionDown(in_channels=512, out_channels=512, kernel_size=3)
        
        #self.fc1 = nn.Linear(16 * 16 * 512, 2)
        self.fc1 = nn.Linear(8 * 8 * 512, 5)
        self.fc2 = nn.Linear(5, 1)

        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """

        #print 'discriminator input shape=', x.shape
        
        x = self.first_conv_layer(x)
        x = self.second_conv_layer(x)
        x = self.third_conv_layer(x)
        x = self.fourth_conv_layer(x)
        x = self.fifth_conv_layer(x)
        x = self.last_conv_layer(x)
        
        x = x.view(BATCH_SIZE, 8 * 8 * 512)
        #x = x.view(-1, 16 * 16 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        sigmoid_out = nn.functional.sigmoid(x)

        return sigmoid_out

###################################################################################

# refer to 'https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/neural_style_transfer/main.py' for vgg feature extraction

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

dtype = torch.cuda.FloatTensor if is_gpu_mode else torch.FloatTensor

# Load image file and convert it into variable (preparing for vgg model)
# unsqueeze for make the 4D tensor to perform conv arithmetic
'''
def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    
    if max_size is not None:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    
    if shape is not None:
        image = image.resize(shape, Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image.type(dtype)
'''

'''
# vgg19 structure
self.vgg._modules.items() = 
[
('0', Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('1', ReLU(inplace)), 
('2', Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('3', ReLU(inplace)), 
('4', MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))), 
('5', Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('6', ReLU(inplace)), 
('7', Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('8', ReLU(inplace)), 
('9', MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))), 
('10', Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('11', ReLU(inplace)), 
('12', Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('13', ReLU(inplace)), 
('14', Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('15', ReLU(inplace)), 
('16', Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('17', ReLU(inplace)), 
('18', MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))), 
('19', Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('20', ReLU(inplace)), 
('21', Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
('22', ReLU(inplace)), 
('23', Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), ('24', ReLU(inplace)), 
('25', Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('26', ReLU(inplace)), 
('27', MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))), 
('28', Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('29', ReLU(inplace)), 
('30', Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('31', ReLU(inplace)), 
('32', Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('33', ReLU(inplace)), 
('34', Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))), 
('35', ReLU(inplace)), 
('36', MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1)))
]

conv_layers = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
'''

# Pretrained VGGNet 
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        
        """Select conv1_1 ~ conv5_1 activation maps."""
        #self.select = ['0', '5', '10', '19', '28'] 
        self.select = ['0','2','5','7','10','12','14','16','19','21','23','25','28','30','32','34'] 
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x):
        """Extract 5 conv activation maps from an input image.
        
        Args:
            x: 4D tensor of shape (1, 3, height, width).
        
        Returns:
            features: a list containing 5 conv activation maps.
        """
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

###################################################################################

if __name__ == "__main__":
    print 'main'

    gen_model = Tiramisu()
    disc_model = Discriminator()
    vgg = VGGNet()
    
    if is_gpu_mode:
        gen_model.cuda()
        disc_model.cuda()
        vgg.cuda()

    optimizer_gen = torch.optim.Adam(gen_model.parameters(), lr=LEARNING_RATE_GENERATOR)
    optimizer_disc = torch.optim.Adam(disc_model.parameters(), lr=LEARNING_RATE_DISCRIMINATOR)

    # read imgs
    image_buff_read_index = 0

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT))
    
    answer_img = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH_ANSWER, data_loader.INPUT_IMAGE_HEIGHT_ANSWER))

    bicubic_out_4x = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH_ANSWER, data_loader.INPUT_IMAGE_HEIGHT_ANSWER))
    
    gen_out = np.empty(shape=(BATCH_SIZE, 3, data_loader.INPUT_IMAGE_WIDTH_ANSWER, data_loader.INPUT_IMAGE_HEIGHT_ANSWER))
    
    # opencv style
    temp_img_opencv = np.empty(shape=(data_loader.INPUT_IMAGE_WIDTH, data_loader.INPUT_IMAGE_HEIGHT, 3))
    
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

            # debug
            #print 'input img=', input_img[j]
            #print 'answer_img=', answer_img[j]
            
            np.copyto(input_img[j], data_loader.input_buff[image_buff_read_index])
            np.copyto(answer_img[j], data_loader.answer_buff[image_buff_read_index])

            data_loader.buff_status[image_buff_read_index] = 'empty'

            image_buff_read_index = image_buff_read_index + 1
            if image_buff_read_index >= data_loader.image_buffer_size:
                image_buff_read_index = 0

        if is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
            answers = Variable(torch.from_numpy(answer_img).float().cuda())
        else:
            inputs = Variable(torch.from_numpy(input_img).float())
            answers = Variable(torch.from_numpy(answer_img).float())

        # feedforward the inputs. generator
        #outputs_gen = gen_model(noise_z)
        outputs_gen = gen_model(inputs)
       
        # feedforward the (input, answer) pairs. discriminator
        output_disc_real = disc_model(answers)
        output_disc_fake = disc_model(outputs_gen)

        # lsgan loss for the discriminator
        loss_disc_total_lsgan = 0.5 * (torch.mean((output_disc_real - 1)**2) + torch.mean(output_disc_fake**2))

        # l1-loss between real and fake
        #l1_loss = F.l1_loss(outputs_gen, answers)

        ###############################################################
        # vgg feature extraction
      
        answers_norm = answers * 0.0157 - 2
        outputs_gen_norm = outputs_gen * 0.0157 - 2
        
        vgg_answer_out = vgg(answers_norm)
        vgg_gen_out = vgg(outputs_gen_norm)
        
        # debug
        #print 'answers_norm=', answers_norm
        #print 'answers=', answers
        
        content_loss = 0
        for f_gen_out, f_answer_out in zip(vgg_gen_out, vgg_answer_out):
            # Compute content loss 
            content_loss += torch.mean((f_gen_out - f_answer_out)**2)
            
        l1_loss_vgg_feature = 0.05 * content_loss
        ###############################################################
        
        # vanilla gan loss for the generator
        #loss_gen_vanilla = F.binary_cross_entropy(output_disc_fake, ones_label)

        # lsgan loss for the generator
        loss_gen_lsgan = 0.5 * torch.mean((output_disc_fake - 1)**2)

        #loss_gen_total_lsgan = 5 * loss_gen_lsgan + 0.01 * l1_loss
        loss_gen_total_lsgan = loss_gen_lsgan + l1_loss_vgg_feature
                  
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

        #if i % 500 == 0:
        if i % 50 == 0:
            print '-----------------------------------------------'
            print '-----------------------------------------------'
            print 'iterations = ', str(i)
            print 'loss(generator)     = ', str(loss_gen_lsgan)
            #print 'loss(l1)     = ', str(l1_loss)
            print 'loss(l1_loss_vgg_feature)     = ', str(l1_loss_vgg_feature)
            print 'loss(discriminator) = ', str(loss_disc_total_lsgan)
            print '-----------------------------------------------'
            print '(discriminator out-real) = ', output_disc_real[0]
            print '(discriminator out-fake) = ', output_disc_fake[0]

            # tf-board (scalar)
            logger.scalar_summary('loss-generator', loss_gen_lsgan, i)
            #logger.scalar_summary('loss-l1', l1_loss, i)
            logger.scalar_summary('l1_loss_vgg_feature', l1_loss_vgg_feature, i)
            logger.scalar_summary('loss-discriminator', loss_disc_total_lsgan, i)
            logger.scalar_summary('disc-out-for-real', output_disc_real[0], i)
            logger.scalar_summary('disc-out-for-fake', output_disc_fake[0], i)

            # tf-board (images - first 1 batches)
            output_imgs_temp = outputs_gen.cpu().data.numpy()[0:1]

            '''
            output_imgs_temp = outputs_gen.cpu().data[0:1]
            output_imgs_temp = output_imgs_temp.squeeze()
            denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            output_imgs_temp = denorm(output_imgs_temp).clamp_(0, 1)
            gen_out[0] = output_imgs_temp * 255
            '''
            
            answer_imgs_temp = answers.cpu().data.numpy()[0:1]
                       
            inputs_temp = inputs.cpu().data.numpy()[0:1]
            # logger.an_image_summary('generated', output_img, i)
            
            temp_img_opencv[:, :, 0] = inputs_temp[0][0, :, :]
            temp_img_opencv[:, :, 1] = inputs_temp[0][1, :, :]
            temp_img_opencv[:, :, 2] = inputs_temp[0][2, :, :]
            
            bicubic_opencv_4x = cv2.resize(temp_img_opencv, (data_loader.INPUT_IMAGE_WIDTH_ANSWER, data_loader.INPUT_IMAGE_HEIGHT_ANSWER), interpolation=cv2.INTER_CUBIC)
            
            bicubic_out_4x[0][0, :, :] = bicubic_opencv_4x[:, :, 0]
            bicubic_out_4x[0][1, :, :] = bicubic_opencv_4x[:, :, 1]
            bicubic_out_4x[0][2, :, :] = bicubic_opencv_4x[:, :, 2]
            
            logger.image_summary('generated', output_imgs_temp, i)
            logger.image_summary('real', answer_imgs_temp, i)
            logger.image_summary('input', inputs_temp, i)
            logger.image_summary('bicubic(opencv)', bicubic_out_4x, i)
            logger.image_summary('input(dup)', inputs_temp, i)

        # save the model
        if i % MODEL_SAVING_FREQUENCY == 0:
            torch.save(gen_model.state_dict(),
                       MODEL_SAVING_DIRECTORY + 'sr_gan_tiramisu_with_vgg_loss_iter_' + str(i) + '.pt')
