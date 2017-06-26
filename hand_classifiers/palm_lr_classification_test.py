import caffe
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import os
import re

test_image_path = '/home1/irteamsu/users/rklee/data/hand_left_right_classifier/hand_lr_classifier_test_images/both_hands_for_caffe/'

os.chdir(test_image_path)

jpg_files = glob.glob( '*.jpg' )
JPG_files = glob.glob( '*.png' )

if __name__ == '__main__':

    GPU_ID = 0
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    
    mean_filename = '/home1/irteamsu/caffe/data/imagenet_hand_lr_classifier/imagenet_mean_hand_lr_classifier.binaryproto'
    
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]

    # Reference Caffenet
    #palm_pretrained = '/home1/irteamsu/caffe/models/bvlc_reference_caffenet/caffenet_train_hand_lr_classification_iter_200000.caffemodel'
    #palm_model_file = '/home1/irteamsu/caffe/models/bvlc_reference_caffenet/deploy_hand_lr_classification.prototxt'

    # Resnet-101
    #palm_pretrained = '/home1/irteamsu/caffe/models/bvlc_resnet/se_train_resnet_hand_lr_classification_fine_iter_200000.caffemodel'
    #palm_model_file = '/home1/irteamsu/caffe/models/bvlc_resnet/ResNet-101-deploy-hand_lr_classifier.prototxt'

    # VGG16
    palm_pretrained = '/home1/irteamsu/caffe/models/bvlc_vggnet/se_train_vggnet_hand_lr_classifier_fine_iter_200000.caffemodel'
    palm_model_file = '/home1/irteamsu/caffe/models/bvlc_vggnet/VGG_ILSVRC_16_layers_deploy_hand_lr_classifier.prototxt'
    
    palm_net = caffe.Classifier(palm_model_file,
                                     palm_pretrained,
                                     mean=mean,
                                     channel_swap=(2, 1, 0),
                                     raw_scale=255,
                                     image_dims=(256,256))
    idx = 0
    count_correct = 0
    count_incorrect = 0

    for jpg_file in jpg_files:
        #file_name = root_path + 'jn/jn_' + jpg_file
        file_name = test_image_path + jpg_file
        print '#######################################################################'
        
        input_image = caffe.io.load_image(file_name)
        #_ = plt.imshow(input_image)

        prediction = palm_net.predict([input_image], oversample=False)
        
        print 'test_idx = ', str(idx)
        print 'input filename: ', file_name
        print 'prediction = ', prediction
        print 'type = ', np.argmax(prediction)
        
        match = re.search("right_2017", file_name)
        if match:
            print 'right_handed input'
            if np.argmax(prediction) == 0:
                count_correct = count_correct + 1
                print 'result: correct'
            else:
                count_incorrect = count_incorrect + 1
                print 'result: incorrect'
        else:
            print 'left_handed input'
            if np.argmax(prediction) == 1:
                count_correct = count_correct + 1
                print 'result: correct'
            else:
                count_incorrect = count_incorrect + 1
                print 'result: incorrect'    
        
        idx = idx + 1

    for JPG_file in JPG_files:
        #file_name = root_path + 'jn/jn_' + JPG_file
        file_name = test_image_path + JPG_file
        print '#######################################################################'

        input_image = caffe.io.load_image(file_name)
        #_ = plt.imshow(input_image)
        prediction = palm_net.predict([input_image], oversample=False)
        
        print 'test_idx = ', str(idx)
        print 'input filename: ', file_name
        print 'prediction = ', prediction
        print 'type = ', np.argmax(prediction)
        
        match = re.search("right_2017", file_name)
        if match:
            print 'right_handed input'
            if np.argmax(prediction) == 0:
                count_correct = count_correct + 1
                print 'result: correct'
            else:
                count_incorrect = count_incorrect + 1
                print 'result: incorrect'
        else:
            print 'left_handed input'
            if np.argmax(prediction) == 1:
                count_correct = count_correct + 1
                print 'result: correct'
            else:
                count_incorrect = count_incorrect + 1
                print 'result: incorrect'    
                
        idx = idx + 1
    
    print 'count_correct', count_correct
    print 'count_incorrect', count_incorrect
    print 'total_test_index', idx
    print 'Accuracy = ', str(100.0 * count_correct / idx)
    print 'End'
