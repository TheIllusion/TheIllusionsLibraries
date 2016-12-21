import tensorflow as tf
import numpy as np
import threading
import re
import time
import cv2
import random

TRAINING_LIST_FILE_NAME = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/shuffle_training_list.txt'
SOURCE_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/training_data/'

#HAND_IMAGE_DIRECTORY = '/media/illusion/ML_Linux/Data/hand_detection/hands'
#NON_HAND_IMAGE_DIRECTORY = '/media/illusion/ML_Linux/Data/hand_detection/non_hands'

BATCH_SIZE = 50

PSEUDO_MEAN_PIXEL_VALUE = 100


##############################################################################################
# Image Buffer Management

# image buffers
image_buffer_size = 150
input_buff = np.empty(shape=(image_buffer_size, 512, 512, 3))
answer_buff = np.empty(shape=(image_buffer_size, 1))

buff_status = []
for i in range(image_buffer_size):
    buff_status.append('empty')

current_buff_index = 0
lineIdx = 0

training_list_file = open(TRAINING_LIST_FILE_NAME)
training_list = training_list_file.readlines()
max_training_index = len(training_list)

def image_buffer_loader():
    global current_buff_index
    global lineIdx

    print 'image_buffer_loader'

    while True:
        filename_ = training_list[lineIdx]

        match = re.search(".jpg", filename_)
        if match:
            end_index = match.end()
            filename = filename_[0:end_index]

        match = re.search(".JPG", filename_)
        if match:
            end_index = match.end()
            filename = filename_[0:end_index]

        match = re.search(".jpeg", filename_)
        if match:
            end_index = match.end()
            filename = filename_[0:end_index]

        match = re.search(".JPEG", filename_)
        if match:
            end_index = match.end()
            filename = filename_[0:end_index]

        match = re.search(".png", filename_)
        if match:
            end_index = match.end()
            filename = filename_[0:end_index]

        match = re.search(".PNG", filename_)

        if match:
            end_index = match.end()
            filename = filename_[0:end_index]

        training_file_name = SOURCE_IMAGE_DIRECTORY + filename
        #answer_file_name = answer_image_directory + filename

        while buff_status[current_buff_index] == 'filled':
            #print 'sleep start'
            time.sleep(1)
            #print 'sleep end'
            if buff_status[current_buff_index] == 'empty':
                break

        input_img = cv2.imread(training_file_name, cv2.IMREAD_COLOR)
        #answer_img = cv2.imread(answer_file_name, cv2.IMREAD_GRAYSCALE)

        if (type(input_img) is not np.ndarray):
            lineIdx = lineIdx + 1
            if lineIdx >= max_training_index:
                lineIdx = 0
            continue

        '''
        if (type(answer_img) is not np.ndarray):
            lineIdx = lineIdx + 1
            if lineIdx >= max_training_index:
                lineIdx = 0
            continue
        '''

        input_buff[current_buff_index] = cv2.resize(input_img, (512,512), interpolation=cv2.INTER_LINEAR)
        classifying_string = filename_[end_index+1]
        answer_buff[current_buff_index] = float(classifying_string)

        buff_status[current_buff_index] = 'filled'

        #pseudo_mean_value = np.mean(input_buff[current_buff_index])
        input_buff[current_buff_index] = input_buff[current_buff_index] - PSEUDO_MEAN_PIXEL_VALUE

        if lineIdx % 200 == 0:
            #print 'mean = ' + str(pseudo_mean_value)
            print 'training_jpg_line_idx=', str(lineIdx)
            #print 'mean_y) = ' + str(y_mean_value)

        lineIdx = lineIdx + 1
        if lineIdx >= max_training_index:
            lineIdx = 0

        #print 'current_buff_index=', current_buff_index

        current_buff_index = current_buff_index + 1
        if current_buff_index >= image_buffer_size:
            current_buff_index = 0

#Launch image buffer loader
timer = threading.Timer(1, image_buffer_loader)
timer.start()

#debug
#image_buffer_loader()
##############################################################################################

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

X = tf.placeholder(tf.float32, [BATCH_SIZE, 512, 512, 3])
Y = tf.placeholder(tf.float32, [BATCH_SIZE, 1])

CONV_W1_HAND = tf.get_variable("CNN_W1_HAND", shape=[5,5,3, 16], initializer=tf.contrib.layers.xavier_initializer())
CONV_W2_HAND = tf.get_variable("CNN_W2_HAND", shape=[5,5,16,32], initializer=tf.contrib.layers.xavier_initializer())
CONV_W3_HAND = tf.get_variable("CNN_W3_HAND", shape=[3,3,32,64], initializer=tf.contrib.layers.xavier_initializer())
CONV_W4_HAND = tf.get_variable("CNN_W4_HAND", shape=[3,3,64,128], initializer=tf.contrib.layers.xavier_initializer())
CONV_W5_HAND = tf.get_variable("CNN_W5_HAND", shape=[3,3,128,256], initializer=tf.contrib.layers.xavier_initializer())
CONV_W6_HAND = tf.get_variable("CNN_W6_HAND", shape=[3,3,256,512], initializer=tf.contrib.layers.xavier_initializer())

FC_W1 = tf.get_variable("FC_W1_HAND", shape = [64*64, 500], initializer=tf.contrib.layers.xavier_initializer())
FC_W2 = tf.get_variable("FC_W2_HAND", shape = [500, 500], initializer=tf.contrib.layers.xavier_initializer())
FC_W3 = tf.get_variable("FC_W3_HAND", shape = [500, 2], initializer=tf.contrib.layers.xavier_initializer())

BIAS_CONV_W1_HAND = tf.Variable(tf.zeros([1,16]), name = "ConvBiasHand1")
BIAS_CONV_W2_HAND = tf.Variable(tf.zeros([1,32]), name = "ConvBiasHand2")
BIAS_CONV_W3_HAND = tf.Variable(tf.zeros([1,64]), name = "ConvBiasHand3")
BIAS_CONV_W4_HAND = tf.Variable(tf.zeros([1,128]), name = "ConvBiasHand4")
BIAS_CONV_W5_HAND = tf.Variable(tf.zeros([1,256]), name = "ConvBiasHand5")
BIAS_CONV_W6_HAND = tf.Variable(tf.zeros([1,512]), name = "ConvBiasHand6")

BIAS_FC_W1 = tf.Variable(tf.zeros([1,500]), name = "FCBiasHand1")
BIAS_FC_W2 = tf.Variable(tf.zeros([1,500]), name = "FCBiasHand2")
BIAS_FC_W3 = tf.Variable(tf.zeros([1,2]), name = "FCBiasHand3")

X_IN = tf.reshape(X, [-1, 512, 512, 3])

#512->256
conv1 = tf.nn.relu(tf.nn.conv2d(X_IN, CONV_W1_HAND, strides=[1,2,2,1], padding = 'SAME') + BIAS_CONV_W1_HAND)
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, CONV_W2_HAND, strides=[1,1,1,1], padding = 'SAME') + BIAS_CONV_W2_HAND)
#256->128
conv2_pool = max_pool_2x2(conv2)
conv3 = tf.nn.relu(tf.nn.conv2d(conv2_pool, CONV_W3_HAND, strides=[1,1,1,1], padding = 'SAME') + BIAS_CONV_W3_HAND)
conv4 = tf.nn.relu(tf.nn.conv2d(conv3, CONV_W4_HAND, strides=[1,1,1,1], padding = 'SAME') + BIAS_CONV_W4_HAND)
conv5 = tf.nn.relu(tf.nn.conv2d(conv4, CONV_W5_HAND, strides=[1,1,1,1], padding = 'SAME') + BIAS_CONV_W5_HAND)
conv6 = tf.nn.relu(tf.nn.conv2d(conv5, CONV_W6_HAND, strides=[1,1,1,1], padding = 'SAME') + BIAS_CONV_W6_HAND)
#128->64
conv6_pool = max_pool_2x2(conv6)

FC_IN = tf.reshape(conv6_pool, [-1, 64 * 64])
FC1 = tf.nn.relu(tf.matmul(FC_IN, FC_W1) + BIAS_FC_W1)
FC2 = tf.nn.relu(tf.matmul(FC1, FC_W2) + BIAS_FC_W2)
FC3 = tf.nn.relu(tf.matmul(FC2, FC_W3) + BIAS_FC_W3)

hypothesis_hand = tf.nn.softmax(FC3)

cost_hand = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis_hand), reduction_indices=1))

#learning rate
learning_rate_hand = tf.Variable(0.001)

optimizer_hand = tf.train.AdamOptimizer(learning_rate_hand)
train_hand = optimizer_hand.minimize(cost_hand)

#dropout rate
dropout_rate_hand = tf.placeholder("float")

init_hand = tf.initialize_all_variables()

image_buff_read_index = 0

with tf.Session() as sess:
    sess.run(init_hand)

    # train the model
    for step in xrange(2001):
        x_data = np.empty(shape=(BATCH_SIZE, 512, 512, 3))
        y_data = np.empty(shape=(BATCH_SIZE, 1))

        for batchIdx in xrange(BATCH_SIZE):
            # read image from buffer
            while buff_status[image_buff_read_index] == 'empty':
                time.sleep(1)
                print 'wait buffers to be filled'
                if buff_status[image_buff_read_index] == 'filled':
                    break

            image_buff_read_index = image_buff_read_index + 1
            if image_buff_read_index >= image_buffer_size:
                image_buff_read_index = 0

        # Deep copy the image buffers
        np.copyto(x_data[batchIdx], input_buff[image_buff_read_index])
        np.copyto(y_data[batchIdx], answer_buff[image_buff_read_index])

        buff_status[image_buff_read_index] = 'empty'

        sess.run(train_hand, feed_dict={X: x_data, Y: y_data, dropout_rate_hand: 0.7})
        if step % 10 == 0:
            print step, sess.run(cost_hand, feed_dict={X: x_data, Y: y_data})
