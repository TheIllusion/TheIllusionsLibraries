import torch
#from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as transforms
from generator_tiramisu import Tiramisu
from discriminator import Discriminator
import time, cv2
import os, glob
import numpy as np
import imageio

INPUT_TEST_IMAGE_WIDTH = 256
INPUT_TEST_IMAGE_HEIGHT = 256

FIXED_GENERATION_LENGTH = 20

# tbt005
INPUT_TEST_IMAGE_DIRECTORY_PATH = "/data/rklee/hair_segmentation/official_test_set/original/"
#INPUT_TEST_IMAGE_DIRECTORY_PATH = "/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/test_first_frames/"

# tbt005 (10.161.31.83)
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/models_3/'
RESULT_JPG_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/generated_test_imgs/'
RESULT_GIF_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/gif_generated_imgs/'

if not os.path.exists(RESULT_JPG_DIRECTORY):
    os.mkdir(RESULT_JPG_DIRECTORY)
    
if not os.path.exists(RESULT_GIF_DIRECTORY):
    os.mkdir(RESULT_GIF_DIRECTORY)
    
# load the filelist
#os.chdir(INPUT_TEST_IMAGE_DIRECTORY_PATH)
jpg_files = glob.glob(INPUT_TEST_IMAGE_DIRECTORY_PATH + '*.jpg')
#random.shuffle(jpg_files)
max_test_index = len(jpg_files)

TEST_SIZE = 53
#TEST_SIZE = 300
#TEST_SIZE = 1

def create_gif(filenames, result_filename, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    #output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(result_filename, images, duration=duration)
    
if __name__ == "__main__":
    
    generator_model = Tiramisu()
        
    generator_model.load_state_dict(torch.load(MODEL_SAVING_DIRECTORY + '2x_cost_video_gans_genenerator_iter_810000.pt'))
    
    # for gpu mode
    generator_model.cuda()
    #generator_model.eval()
    
    # pytorch style
    input_img = np.empty(shape=(1, 3, INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT))
    
    motion_vec_img = np.empty(shape=(1, 1, INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT))
    
    motion_vector = np.empty(shape=(1, INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT))
    
    # opencv style
    output_img_opencv = np.empty(shape=(INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT, 3))
    
    for idx in range(TEST_SIZE):
        # load a single img
        img_opencv = cv2.imread(jpg_files[idx], cv2.IMREAD_COLOR)

        img_opencv = cv2.resize(img_opencv, (INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # bgr to rgb
        input_img_tmp = img_opencv[..., [2,1,0]]
        
        input_img[0][0, :, :] = input_img_tmp[:, :, 0]
        input_img[0][1, :, :] = input_img_tmp[:, :, 1]
        input_img[0][2, :, :] = input_img_tmp[:, :, 2]    
        
        # force gpu mode
        inputs = Variable(torch.from_numpy(input_img).float().cuda())
        #motion_vec = Variable(torch.from_numpy(motion_vec_img).float().cuda())
        
        saved_filelist = []
        
        for loop_idx in range(1,FIXED_GENERATION_LENGTH + 1):
            mv_idx = loop_idx * 10
            motion_vector[:,:,:] = mv_idx
            motion_vec_img[0] = motion_vector
            
            # gpu
            motion_vec = Variable(torch.from_numpy(motion_vec_img).float().cuda())

            start_time = time.time()
            
            # concat
            inputs_with_mv = torch.cat((inputs, motion_vec), 1)
            outputs_gen = generator_model(inputs_with_mv)

            elapsed_time = time.time() - start_time

            print 'elapsed time for processing ' + str(idx) + 'th img: ', str(elapsed_time)

            output_img = outputs_gen.cpu().data.numpy()[0]

            # output_img = output_img * 255

            output_img_opencv[:, :, 0] = output_img[0, :, :]
            output_img_opencv[:, :, 1] = output_img[1, :, :]
            output_img_opencv[:, :, 2] = output_img[2, :, :]

            output_img_opencv_bgr = output_img_opencv[..., [2,1,0]]
            
            cv2.imwrite(RESULT_JPG_DIRECTORY + 'result_file_' + str(idx) + '_mv_' + str(loop_idx) +'.jpg', output_img_opencv_bgr)
            
            # preparation for creating a gif file
            saved_filelist.append(RESULT_JPG_DIRECTORY + 'result_file_' + str(idx) + '_mv_' + str(loop_idx) +'.jpg')
            
            # display purposes only. create concatenated imgs (original:feedforward)
            concated_img = np.hstack((img_opencv, output_img_opencv_bgr))
            cv2.imwrite(RESULT_JPG_DIRECTORY + 'concat_file_' + str(idx) + '_mv_' + str(loop_idx) +'.jpg', concated_img)
            
        duration = 0.3
        result_filename = RESULT_GIF_DIRECTORY + 'result_file_' + str(idx) + '_' + str(loop_idx) +'.gif'
        create_gif(saved_filelist, result_filename, duration)
    
    print 'feedforward_img.py main'