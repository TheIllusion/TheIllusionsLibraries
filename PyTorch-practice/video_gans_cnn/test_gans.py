import torch
#from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as transforms
from generator_tiramisu import Tiramisu
from discriminator import Discriminator, BATCH_SIZE
import time, cv2
import os, glob
import numpy as np

INPUT_TEST_IMAGE_WIDTH = 256
INPUT_TEST_IMAGE_HEIGHT = 256

FIXED_GENERATION_LENGTH = 20

# tbt005
INPUT_TEST_IMAGE_DIRECTORY_PATH = "/data/rklee/hair_segmentation/official_test_set/original/"

# tbt005 (10.161.31.83)
MODEL_SAVING_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/models_3/'
RESULT_JPG_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/video_gans_cnn/generated_test_imgs/'

if not os.path.exists(RESULT_JPG_DIRECTORY):
    os.mkdir(RESULT_JPG_DIRECTORY)
    
# load the filelist
#os.chdir(INPUT_TEST_IMAGE_DIRECTORY_PATH)
jpg_files = glob.glob(INPUT_TEST_IMAGE_DIRECTORY_PATH + '*.jpg')
#random.shuffle(jpg_files)
max_test_index = len(jpg_files)

TEST_SIZE = 53
#TEST_SIZE = 300

if __name__ == "__main__":
    
    generator_model = Tiramisu()
    
    # for gpu mode
    generator_model.cuda()

    generator_model.load_state_dict(torch.load(MODEL_SAVING_DIRECTORY + '2x_cost_video_gans_genenerator_iter_710000.pt'))
    
    # pytorch style
    input_img = np.empty(shape=(1, 3, INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT))
    
    motion_vec_img = np.empty(shape=(BATCH_SIZE, 1, INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT))
    
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
        motion_vec = Variable(torch.from_numpy(motion_vec_img).float().cuda())

        start_time = time.time()
        
        for loop_idx in range(1,FIXED_GENERATION_LENGTH + 1):
            mv_idx = loop_idx * 10
            motion_vec[:,:,:] = mv_idx

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

            output_img_opencv = output_img_opencv[..., [2,1,0]]
            
            # display purposes only. create concatenated imgs (original:feedforward)
            concated_img = np.hstack((img_opencv, output_img_opencv))
            cv2.imwrite(RESULT_JPG_DIRECTORY + 'concat_file_' + str(idx) + '_mv_' + str(loop_idx) +'.jpg', concated_img)
    
    print 'feedforward_img.py main'