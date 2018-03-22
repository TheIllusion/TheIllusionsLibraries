import torch
#import tiramisu
#from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as transforms
from tiramisu_model import Tiramisu
import time, cv2
import os, glob
import numpy as np

# tbt005
#INPUT_TEST_IMAGE_DIRECTORY_PATH = "/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/cyclegan_for_unified_hair_dyeing/korean_testA/"

INPUT_TEST_IMAGE_DIRECTORY_PATH = "/data/rklee/sr/Flickr2K/Flickr2K_LR_bicubic/X4/"

RESULT_IMAGE_DIRECTORY_PATH = "/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/forward_imgs_simple_cyclegan/"

MODEL_SAVING_DIRECTORY_PATH = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/GANs/models/'

CHECKPOINT_FILENAME = 'cycle_gen_model_iter_80000.pt'

INPUT_TEST_IMAGE_WIDTH = 256
INPUT_TEST_IMAGE_HEIGHT = 256

ANSWER_IMAGE_WIDTH = INPUT_TEST_IMAGE_WIDTH * 1
ANSWER_IMAGE_HEIGHT = INPUT_TEST_IMAGE_HEIGHT * 1

#TEST_SIZE = 53
#TEST_SIZE = 406
TEST_SIZE = 30

# support batch_size == 1 only
BATCH_SIZE = 1

if __name__ == "__main__":
    
    print 'feedforward_unified_cyclegan.py __main__'
    
    if not os.path.exists(RESULT_IMAGE_DIRECTORY_PATH):
        os.mkdir(RESULT_IMAGE_DIRECTORY_PATH)
        
    # load the filelist
    #os.chdir(INPUT_TEST_IMAGE_DIRECTORY_PATH)
    jpg_files = glob.glob(INPUT_TEST_IMAGE_DIRECTORY_PATH + '*.png')
    #random.shuffle(jpg_files)
    
    max_test_index = len(jpg_files)
    
    tiramisu_gen_model = Tiramisu()
   
    tiramisu_gen_model.load_state_dict(torch.load(MODEL_SAVING_DIRECTORY_PATH + CHECKPOINT_FILENAME))

    tiramisu_gen_model.cuda()
    tiramisu_gen_model.eval()

    # pytorch style
    input_img = np.empty(shape=(BATCH_SIZE, 3, INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT))
    
    # opencv style
    output_img_opencv = np.empty(shape=(ANSWER_IMAGE_WIDTH, ANSWER_IMAGE_HEIGHT, 3))
    
    for idx in range(TEST_SIZE):
        # load a single img
        img_opencv = cv2.imread(jpg_files[idx], cv2.IMREAD_COLOR)
        
        print 'process file:', jpg_files[idx]

        img_opencv = cv2.resize(img_opencv, (INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        img_opencv_ori = img_opencv.copy()
        
        img_opencv = img_opencv[..., [2,1,0]]
        
        input_img[0][0, :, :] = img_opencv[:, :, 0]
        input_img[0][1, :, :] = img_opencv[:, :, 1]
        input_img[0][2, :, :] = img_opencv[:, :, 2]    

        # test purposes only
        '''
        output_img_opencv[:, :, 0] = input_img[0][0, :, :]
        output_img_opencv[:, :, 1] = input_img[0][1, :, :]
        output_img_opencv[:, :, 2] = input_img[0][2, :, :]

        cv2.imwrite("input_check_" + str(idx) + ".jpg", output_img_opencv)
        '''

        inputs = Variable(torch.from_numpy(input_img).float().cuda())

        start_time = time.time()

        #print condition_vectors

        outputs = tiramisu_gen_model(inputs)

        elapsed_time = time.time() - start_time

        print 'elapsed time for processing ' + str(idx) + 'th img: ', str(elapsed_time)

        output_img = outputs.cpu().data.numpy()[0]

        #output_img = output_img * 255

        output_img_opencv[:, :, 0] = output_img[0, :, :]
        output_img_opencv[:, :, 1] = output_img[1, :, :]
        output_img_opencv[:, :, 2] = output_img[2, :, :]

        output_img_opencv = output_img_opencv[..., [2,1,0]]

        #cv2.imwrite(RESULT_IMAGE_DIRECTORY_PATH + os.path.basename(jpg_files[idx]), output_img_opencv)
       
        # display purposes only. create concatenated imgs (original:feedforward)
        concated_img = np.hstack((img_opencv_ori, output_img_opencv))
        cv2.imwrite(RESULT_IMAGE_DIRECTORY_PATH + 'concat_' + str(idx) + '.jpg', concated_img)
    
    print 'process done'
