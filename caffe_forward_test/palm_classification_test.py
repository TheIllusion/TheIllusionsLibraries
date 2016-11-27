import caffe
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import os
import re

test_file_path = '/media/illusion/ML_DATA_HDD/Data/hand_detection/test_set/'

os.chdir(test_file_path)

jpg_files = glob.glob( '*.jpg' )
JPG_files = glob.glob( '*.JPG' )

if __name__ == '__main__':

    '''
    GPU_ID = 0
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    '''

    mean_filename = '/home/illusion/caffe/data/palm_classification_caffenet/palm_classification_mean.binaryproto'
    
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]

    # Reference Caffenet
    #palm_pretrained = '/media/illusion/ML_DATA_HDD/caffe_model_snapshots/bvlc_reference_caffenet/caffenet_train_palm_classification_iter_10000.caffemodel'
    #palm_model_file = '/home/illusion/caffe/models/bvlc_reference_caffenet/deploy_palm_classification.prototxt'

    #palm_pretrained = '/home/illusion/caffe/models/bvlc_reference_caffenet/caffenet_train_palm_classification_iter_50000.caffemodel'
    #palm_model_file = '/home/illusion/caffe/models/bvlc_reference_caffenet/deploy_palm_classification.prototxt'

    # Resnet-101
    #palm_pretrained = '/home/illusion/caffe/models/resnet/se_train_resnet101_palm_classification_fine_iter_50000.caffemodel'
    #palm_model_file = '/home/illusion/caffe/models/resnet/ResNet-101-deploy_palm_classification.prototxt'

    # VGG16
    palm_pretrained = '/media/illusion/ML_DATA_HDD/caffe_model_snapshots/bvlc_vggnet/se_train_vggnet_fine_palm_classification_iter_30000.caffemodel'
    palm_model_file = '/home/illusion/caffe/models/bvlc_vggnet/VGG_ILSVRC_16_layers_deploy_palm_classification.prototxt'

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
        file_name = test_file_path + jpg_file
        print '#######################################################################'
        
        input_image = caffe.io.load_image(file_name)

        prediction = palm_net.predict([input_image])
        
        print 'test_idx = ', str(idx)
        print 'input filename: ', file_name
        print 'prediction = ', prediction
        print 'type = ', np.argmax(prediction)

        match = re.search("non_hand", jpg_file)
        if match:
            if np.argmax(prediction) == 0:
                count_correct = count_correct + 1
            else:
                count_incorrect = count_incorrect + 1
        else:
            if np.argmax(prediction) == 1:
                count_correct = count_correct + 1
            else:
                count_incorrect = count_incorrect + 1

        idx = idx + 1

    for JPG_file in JPG_files:
        file_name = test_file_path + JPG_file
        print '#######################################################################'

        input_image = caffe.io.load_image(file_name)
        #_ = plt.imshow(input_image)
        prediction = palm_net.predict([input_image])
        
        print 'test_idx = ', str(idx)
        print 'input filename: ', file_name
        print 'prediction = ', prediction
        print 'type = ', np.argmax(prediction)

        match = re.search("non_hand", jpg_file)
        if match:
            if np.argmax(prediction) == 0:
                count_correct = count_correct + 1
            else:
                count_incorrect = count_incorrect + 1
        else:
            if np.argmax(prediction) == 1:
                count_correct = count_correct + 1
            else:
                count_incorrect = count_incorrect + 1
        
        idx = idx + 1

    print 'count_correct', count_correct
    print 'count_incorrect', count_incorrect
    print 'total_test_index', idx
    print 'Accuracy = ', str(100.0 * count_correct / idx)
    print 'End'
