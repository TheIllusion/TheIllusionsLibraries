import caffe
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
import os

#root_path = '/home/nhnent/H1/users/rklee/Data/palm_data/test_set/crop_resize_512_512/feed_forward_result/'
root_path = '/home/nhnent/H1/users/rklee/Data/palm_data/test_set/crop_resize_512_512/grayscale/'

os.chdir(root_path)

jpg_files = glob.glob( '*.jpg' )
JPG_files = glob.glob( '*.JPG' )

if __name__ == '__main__':

    GPU_ID = 0
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)

    model_root = "/home/nhnent/H1/users/rklee/palm_type_recognition_test"
    
    #mean_filename = model_root + '/palm_sm/palm_mean.binaryproto'
    #mean_filename = model_root + '/palm_gj/palm_mean.binaryproto'
    #mean_filename = model_root + '/palm_jn/palm_mean.binaryproto'

    #mean_filename = model_root + '/palm_sm_e2e/palm_mean.binaryproto'
    #mean_filename = model_root + '/palm_gj_e2e/palm_mean.binaryproto'
    mean_filename = model_root + '/palm_jn_e2e/palm_mean.binaryproto'
    
    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]

    #palm_pretrained = model_root + '/se_train_resnet_palm_sm_fine_iter_300000.caffemodel'
    #palm_pretrained = model_root + '/se_train_resnet_palm_gj_fine_iter_270000.caffemodel'
    #palm_pretrained = model_root + '/se_train_resnet_palm_jn_fine_iter_270000.caffemodel'
        
    #palm_pretrained = model_root + '/se_train_resnet_palm_sm_e2e_fine_iter_300000.caffemodel'
    #palm_pretrained = model_root + '/se_train_resnet_palm_gj_e2e_fine_iter_300000.caffemodel'
    palm_pretrained = model_root + '/se_train_resnet_palm_jn_e2e_fine_iter_300000.caffemodel'
    
    palm_model_file = model_root + '/deploy.prototxt.resnet.101'

    palm_net = caffe.Classifier(palm_model_file,
                                     palm_pretrained,
                                     mean=mean,
                                     channel_swap=(2, 1, 0),
                                     raw_scale=255,
                                     image_dims=(256,256))
    idx = 0

    for jpg_file in jpg_files:
        #file_name = root_path + 'jn/jn_' + jpg_file
        file_name = root_path + jpg_file
        print '#######################################################################'
        
        input_image = caffe.io.load_image(file_name)
        #_ = plt.imshow(input_image)

        prediction = palm_net.predict([input_image])
        
        print 'test_idx = ', str(idx)
        print 'input filename: ', file_name
        print 'prediction = ', prediction
        print 'type = ', np.argmax(prediction)
        
        idx = idx + 1

    for JPG_file in JPG_files:
        #file_name = root_path + 'jn/jn_' + JPG_file
        file_name = root_path + JPG_file
        print '#######################################################################'

        input_image = caffe.io.load_image(file_name)
        #_ = plt.imshow(input_image)
        prediction = palm_net.predict([input_image])
        
        print 'test_idx = ', str(idx)
        print 'input filename: ', file_name
        print 'prediction = ', prediction
        print 'type = ', np.argmax(prediction)
        
        idx = idx + 1

    print 'End'
