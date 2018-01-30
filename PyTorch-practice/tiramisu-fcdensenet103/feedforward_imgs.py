import torch
import tiramisu
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as transforms
from sub_modules import Layer, TransitionUp, TransitionDown, DenseBlock
import time, cv2
import os, glob
import numpy as np

ENABLE_MODEL_ENSEMBLES = True

# svc003
#INPUT_TEST_IMAGE_DIRECTORY_PATH = "/home1/irteamsu/users/rklee/data_6T/data/hair_semantic_segmentation/until_2017_0823/original_all/"

# tbt005
INPUT_TEST_IMAGE_DIRECTORY_PATH = "/data/rklee/hair_segmentation/official_test_set/original/"
#INPUT_TEST_IMAGE_DIRECTORY_PATH = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/unet/resize_ori_384/'
#INPUT_TEST_IMAGE_DIRECTORY_PATH = '/home1/irteamsu/rklee/TheIllusionsLibraries/PyTorch-practice/tiramisu-fcdensenet103/face_crop/'

# load the filelist
#os.chdir(INPUT_TEST_IMAGE_DIRECTORY_PATH)
jpg_files = glob.glob(INPUT_TEST_IMAGE_DIRECTORY_PATH + '*.jpg')
#random.shuffle(jpg_files)
max_test_index = len(jpg_files)

INPUT_TEST_IMAGE_WIDTH = 384
INPUT_TEST_IMAGE_HEIGHT = 384

TEST_SIZE = 53
#TEST_SIZE = 300

if __name__ == "__main__":
    
    tiramisu_model = tiramisu.Tiramisu()
    
    if ENABLE_MODEL_ENSEMBLES:
        tiramisu_model_2 = tiramisu.Tiramisu()
        tiramisu_model_3 = tiramisu.Tiramisu()
    
    if tiramisu.is_gpu_mode:
        tiramisu_model.cuda()

        if ENABLE_MODEL_ENSEMBLES:
            tiramisu_model_2.cuda()
            tiramisu_model_3.cuda()

    tiramisu_model.load_state_dict(torch.load(tiramisu.MODEL_SAVING_DIRECTORY + 'tiramisu_lfw_added_zero_centr_lr_0_0003_iter_960000.pt'))

    if ENABLE_MODEL_ENSEMBLES:
        tiramisu_model_2.load_state_dict(torch.load(tiramisu.MODEL_SAVING_DIRECTORY + 'tiramisu_lfw_added_zero_centr_lr_0_0001_iter_960000.pt'))

        tiramisu_model_3.load_state_dict(torch.load(tiramisu.MODEL_SAVING_DIRECTORY + 'tiramisu_lfw_added_zero_centr_lr_0_0003_iter_990000.pt'))
    
    # pytorch style
    input_img = np.empty(shape=(1, 3, INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT))
    
    # opencv style
    output_img_opencv = np.empty(shape=(INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT, 3))
    
    for idx in range(TEST_SIZE):
        # load a single img
        img_opencv = cv2.imread(jpg_files[idx], cv2.IMREAD_COLOR)

        img_opencv = cv2.resize(img_opencv, (INPUT_TEST_IMAGE_WIDTH, INPUT_TEST_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        input_img[0][0, :, :] = img_opencv[:, :, 0]
        input_img[0][1, :, :] = img_opencv[:, :, 1]
        input_img[0][2, :, :] = img_opencv[:, :, 2]    
        
        # zero-centered input
        input_img[0] = input_img[0] - tiramisu.MEAN_VALUE_FOR_ZERO_CENTERED
        
        # test purposes only
        '''
        output_img_opencv[:, :, 0] = input_img[0][0, :, :]
        output_img_opencv[:, :, 1] = input_img[0][1, :, :]
        output_img_opencv[:, :, 2] = input_img[0][2, :, :]
        
        cv2.imwrite("input_check_" + str(idx) + ".jpg", output_img_opencv)
        '''
        
        if tiramisu.is_gpu_mode:
            inputs = Variable(torch.from_numpy(input_img).float().cuda())
        else:
            inputs = Variable(torch.from_numpy(input_img).float())

        start_time = time.time()
        
        outputs = tiramisu_model(inputs)
        if ENABLE_MODEL_ENSEMBLES:
            outputs_2 = tiramisu_model_2(inputs)
            outputs_3 = tiramisu_model_3(inputs)
        
        elapsed_time = time.time() - start_time
        
        print 'elapsed time for processing ' + str(idx) + 'th img: ', str(elapsed_time)

        output_img = outputs.cpu().data.numpy()[0]
        if ENABLE_MODEL_ENSEMBLES:
            output_img_2 = outputs_2.cpu().data.numpy()[0]
            output_img_3 = outputs_3.cpu().data.numpy()[0]
        
        if not ENABLE_MODEL_ENSEMBLES:
            output_img = output_img * 255
        else:
            #output_img = (output_img + output_img_2) * 255 / 2
            output_img = (output_img + output_img_2 + output_img_3) * 255 / 3

        output_img_opencv[:, :, 0] = output_img[0, :, :]
        output_img_opencv[:, :, 1] = output_img[1, :, :]
        output_img_opencv[:, :, 2] = output_img[2, :, :]

        cv2.imwrite(os.path.basename(jpg_files[idx]), output_img_opencv)
        
        # display purposes only. create concatenated imgs (original:feedforward)
        concated_img = np.hstack((img_opencv, output_img_opencv))
        cv2.imwrite('concat_' + str(idx) + '.jpg', concated_img)
    
    print 'feedforward_img.py main'