import tensorflow as tf
import numpy as np
import threading
import re
import time
import cv2
import random
import os
import glob

IS_TRAINING = True

TRAINED_MODEL_NAME = 'the_simplest_hand_classifier_v2_16000.ckpt'

# Macbook Pro
#ROOT_DIRECTORY = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/'

# Vincent
# ROOT_DIRECTORY = '/home/nhnent/H2/users/rklee/the_simplest_hand_classifier_v1/'

# SVC002
ROOT_DIRECTORY = '/data/users/rklee/hand_classifier/'

# Macbook Pro
# TRAINING_LIST_FILE_NAME = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/shuffle_training_list.txt'
# SOURCE_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/training_data/'

# Vincent
# TRAINING_LIST_FILE_NAME = '/home/nhnent/H2/users/rklee/the_simplest_hand_classifier_v1/shuffle_training_list.txt'
# SOURCE_IMAGE_DIRECTORY = '/home/nhnent/H2/users/rklee/the_simplest_hand_classifier_v1/training_data/'
# TEST_IMAGE_DIRECTORY = '/home/nhnent/H2/users/rklee/the_simplest_hand_classifier_v1/test_sets/test_set/'

# SVC003
TRAINING_LIST_FILE_NAME = ROOT_DIRECTORY + 'shuffle_training_list.txt'
SOURCE_IMAGE_DIRECTORY = ROOT_DIRECTORY + 'training_data/'
TEST_IMAGE_DIRECTORY = ROOT_DIRECTORY + 'test_set/'

PSEUDO_MEAN_PIXEL_VALUE = 100

LEARNING_RATE = 0.00001

if IS_TRAINING:
    BATCH_SIZE = 20
else:
    BATCH_SIZE = 1

##############################################################################################
# Load Test Image lists
os.chdir(TEST_IMAGE_DIRECTORY)

test_jpg_files = glob.glob('*.jpg')
test_JPG_files = glob.glob('*.JPG')

##############################################################################################
# Image Buffer Management

# image buffers
image_buffer_size = 200
input_buff = np.empty(shape=(image_buffer_size, 512, 512, 3))
# answer_buff = np.empty(shape=(image_buffer_size, 1))
answer_buff = np.empty(shape=(image_buffer_size))

buff_status = []
for i in range(image_buffer_size):
    buff_status.append('empty')

current_buff_index = 0
lineIdx = 0

training_list_file = open(TRAINING_LIST_FILE_NAME)
training_list = training_list_file.readlines()
max_training_index = len(training_list)

exit_notification = False


def image_buffer_loader():
    global current_buff_index
    global lineIdx

    print 'image_buffer_loader'

    while True:
        filename_ = training_list[lineIdx]

        end_index = 0

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
        
	if end_index == 0:
            lineIdx = lineIdx + 1
            if lineIdx >= max_training_index:
                lineIdx = 0

            print 'skip this jpg file. continue.'
            continue

        training_file_name = SOURCE_IMAGE_DIRECTORY + filename
        # answer_file_name = answer_image_directory + filename

        while buff_status[current_buff_index] == 'filled':
            if exit_notification == True:
                break

            # print 'sleep start'
            time.sleep(1)
            # print 'sleep end'
            if buff_status[current_buff_index] == 'empty':
                break

        if exit_notification == True:
            break

        input_img = cv2.imread(training_file_name, cv2.IMREAD_COLOR)
        # answer_img = cv2.imread(answer_file_name, cv2.IMREAD_GRAYSCALE)

        if (type(input_img) is not np.ndarray):
            lineIdx = lineIdx + 1
            if lineIdx >= max_training_index:
                lineIdx = 0

            print 'skip this jpg file. continue.'
            continue

        input_buff[current_buff_index] = cv2.resize(input_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        classifying_string = filename_[end_index + 1]

        # answer_buff[current_buff_index][0] = float(classifying_string)
        answer_buff[current_buff_index] = float(classifying_string)

        buff_status[current_buff_index] = 'filled'

        # pseudo_mean_value = np.mean(input_buff[current_buff_index])
        input_buff[current_buff_index] = input_buff[current_buff_index] - PSEUDO_MEAN_PIXEL_VALUE

        if lineIdx % 200 == 0:
            # print 'mean = ' + str(pseudo_mean_value)
            print 'training_jpg_line_idx=', str(lineIdx)
            # print 'mean_y) = ' + str(y_mean_value)

        lineIdx = lineIdx + 1
        if lineIdx >= max_training_index:
            lineIdx = 0

        # print 'current_buff_index=', current_buff_index

        current_buff_index = current_buff_index + 1
        if current_buff_index >= image_buffer_size:
            current_buff_index = 0


# Launch image buffer loader
if IS_TRAINING:
    timer = threading.Timer(1, image_buffer_loader)
    timer.start()


# debug
# image_buffer_loader()
##############################################################################################

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


X = tf.placeholder(tf.float32, [BATCH_SIZE, 512, 512, 3])
Y = tf.placeholder(tf.float32, [BATCH_SIZE, 1])

CONV_W1_HAND = tf.get_variable("CNN_W1_HAND", shape=[11, 11, 3, 48], initializer=tf.contrib.layers.xavier_initializer())
CONV_W2_HAND = tf.get_variable("CNN_W2_HAND", shape=[11, 11, 48, 48], initializer=tf.contrib.layers.xavier_initializer())
CONV_W3_HAND = tf.get_variable("CNN_W3_HAND", shape=[5, 5, 48, 64], initializer=tf.contrib.layers.xavier_initializer())
CONV_W4_HAND = tf.get_variable("CNN_W4_HAND", shape=[3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
CONV_W5_HAND = tf.get_variable("CNN_W5_HAND", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
CONV_W6_HAND = tf.get_variable("CNN_W6_HAND", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())

FC_W1 = tf.get_variable("FC_W1_HAND", shape=[16 * 16 * 128, 1000], initializer=tf.contrib.layers.xavier_initializer())
FC_W2 = tf.get_variable("FC_W2_HAND", shape=[1000, 1000], initializer=tf.contrib.layers.xavier_initializer())
FC_W3 = tf.get_variable("FC_W3_HAND", shape=[1000, 1], initializer=tf.contrib.layers.xavier_initializer())

BIAS_CONV_W1_HAND = tf.Variable(tf.zeros([1, 48]), name="ConvBiasHand1")
BIAS_CONV_W2_HAND = tf.Variable(tf.zeros([1, 48]), name="ConvBiasHand2")
BIAS_CONV_W3_HAND = tf.Variable(tf.zeros([1, 64]), name="ConvBiasHand3")
BIAS_CONV_W4_HAND = tf.Variable(tf.zeros([1, 64]), name="ConvBiasHand4")
BIAS_CONV_W5_HAND = tf.Variable(tf.zeros([1, 128]), name="ConvBiasHand5")
BIAS_CONV_W6_HAND = tf.Variable(tf.zeros([1, 128]), name="ConvBiasHand6")

BIAS_FC_W1 = tf.Variable(tf.zeros([1, 1000]), name="FCBiasHand1")
BIAS_FC_W2 = tf.Variable(tf.zeros([1, 1000]), name="FCBiasHand2")
BIAS_FC_W3 = tf.Variable(tf.zeros([1, 1]), name="FCBiasHand3")

X_IN = tf.reshape(X, [-1, 512, 512, 3])

# 512->256
conv1 = tf.nn.relu(tf.nn.conv2d(X_IN, CONV_W1_HAND, strides=[1, 2, 2, 1], padding='SAME') + BIAS_CONV_W1_HAND)
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, CONV_W2_HAND, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W2_HAND)
# 256->128
conv2_pool = max_pool_2x2(conv2)
conv3 = tf.nn.relu(tf.nn.conv2d(conv2_pool, CONV_W3_HAND, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W3_HAND)
# 128->64
conv3_pool = max_pool_2x2(conv3)
conv4 = tf.nn.relu(tf.nn.conv2d(conv3_pool, CONV_W4_HAND, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W4_HAND)
# 64->32
conv4_pool = max_pool_2x2(conv4)
conv5 = tf.nn.relu(tf.nn.conv2d(conv4_pool, CONV_W5_HAND, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W5_HAND)
conv6 = tf.nn.relu(tf.nn.conv2d(conv5, CONV_W6_HAND, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W6_HAND)
# 32->16
conv6_pool = max_pool_2x2(conv6)

FC_IN = tf.reshape(conv6_pool, [-1, 16 * 16 * 128])
FC1 = tf.nn.relu(tf.matmul(FC_IN, FC_W1) + BIAS_FC_W1)
FC2 = tf.nn.relu(tf.matmul(FC1, FC_W2) + BIAS_FC_W2)

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
FC2_DROPOUT = tf.nn.dropout(FC2, keep_prob)

# hypothesis_hand = tf.nn.softmax(tf.matmul(FC2, FC_W3) + BIAS_FC_W3)
hypothesis_hand = tf.nn.sigmoid(tf.matmul(FC2_DROPOUT, FC_W3) + BIAS_FC_W3)

if IS_TRAINING:
    regularization_term = 0.000005 * (tf.nn.l2_loss(CONV_W1_HAND)
                                      + tf.nn.l2_loss(CONV_W2_HAND)
                                      + tf.nn.l2_loss(CONV_W3_HAND)
                                      + tf.nn.l2_loss(CONV_W4_HAND)
                                      + tf.nn.l2_loss(CONV_W5_HAND)
                                      + tf.nn.l2_loss(CONV_W6_HAND)
                                      + tf.nn.l2_loss(FC_W1)
                                      + tf.nn.l2_loss(FC_W2)
                                      + tf.nn.l2_loss(FC_W3))

    # cost_hand = tf.reduce_mean( (-tf.reduce_sum(Y*tf.log(hypothesis_hand) , reduction_indices=1)) + regularization_term )
    # cost_hand = tf.reduce_mean( -tf.reduce_sum(Y*tf.log(hypothesis_hand)) + regularization_term )

    cost_hand = (-tf.reduce_mean(Y * tf.log(tf.clip_by_value(hypothesis_hand, 1e-30, 1.0))
                                 + (1 - Y) * tf.log(tf.clip_by_value(1 - hypothesis_hand, 1e-30, 1.0)))
                 + regularization_term)

    '''
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y, hypothesis_hand))
    cost_hand = cross_entropy
    '''

    # learning rate
    learning_rate_hand = tf.Variable(LEARNING_RATE)

    optimizer_hand = tf.train.AdamOptimizer(learning_rate_hand)
    train_hand = optimizer_hand.minimize(cost_hand)

    # dropout rate
    dropout_rate_hand = tf.placeholder("float")

    init_hand = tf.initialize_all_variables()

# tf.global_variables_initializer()

image_buff_read_index = 0

##############################################################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    if IS_TRAINING:
        sess.run(init_hand)

        # train the model
        for step in xrange(500001):
            x_data = np.empty(shape=(BATCH_SIZE, 512, 512, 3))
            y_data = np.empty(shape=(BATCH_SIZE, 1))

            for batchIdx in xrange(BATCH_SIZE):
                # read image from buffer
                while buff_status[image_buff_read_index] == 'empty':
                    time.sleep(1)
                    print 'wait buffers to be filled'
                    if buff_status[image_buff_read_index] == 'filled':
                        break

                # Deep copy the image buffers
                np.copyto(x_data[batchIdx], input_buff[image_buff_read_index])
                np.copyto(y_data[batchIdx], answer_buff[image_buff_read_index])
                buff_status[image_buff_read_index] = 'empty'

                image_buff_read_index = image_buff_read_index + 1
                if image_buff_read_index >= image_buffer_size:
                    image_buff_read_index = 0

            sess.run(train_hand, feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
            # sess.run(train_hand, feed_dict={X: x_data, Y: y_data, dropout_rate_hand: 0.7})
            if step % 50 == 0:
                print 'step = ', step
                print 'regularization cost = ', sess.run(regularization_term)
                print 'total cost = ', sess.run(cost_hand, feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
                print 'hypothesis = ', sess.run(hypothesis_hand, feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
                print '-------------------------------'

            if step % 5000 == 0:
                saver = tf.train.Saver(tf.trainable_variables())
                file_name = ROOT_DIRECTORY + "models/the_simplest_hand_classifier_v2_" + str(step) + ".ckpt"
                save_path = saver.save(sess, file_name)
                print("Model saved in file: %s" % save_path)

    # Feed-forward Test
    else:
        # Load the pre-trained model from file
        saver = tf.train.Saver()
        saver.restore(sess, ROOT_DIRECTORY + "models/" + TRAINED_MODEL_NAME)

        # sess.run(init_hand)

        # reader = tf.train.NewCheckpointReader(ROOT_DIRECTORY + 'models/' + TRAINED_MODEL_NAME)

        print("Model restored.")

    exit_notification = True

    ##############################################################################################
    # Test the trained model
    # Read the test data
    if not IS_TRAINING:
        idx = 0
        count_correct = 0
        count_incorrect = 0

        x_test_data = np.empty(shape=(1, 512, 512, 3))

        for jpg_file in test_jpg_files:
            file_name = TEST_IMAGE_DIRECTORY + jpg_file
            print '#######################################################################'

            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            input_image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

            network_output = tf.to_float(hypothesis_hand)
            print 'after mean subtraction. PSEUDO_MEAN = ', PSEUDO_MEAN_PIXEL_VALUE
            x_test_data[0] = input_image
            x_test_data = x_test_data - PSEUDO_MEAN_PIXEL_VALUE
            # print x_test_data
            # print input_image
            prediction = sess.run([network_output], feed_dict={X: x_test_data})

            print 'test_idx = ', str(idx)
            print 'input filename: ', file_name
            print 'prediction = ', prediction
            # print 'type = ', np.argmax(prediction)

            match = re.search("non_hand", jpg_file)

            if match:
                # if np.argmax(prediction) == 0:
                if prediction[0] < 0.5:
                    count_correct = count_correct + 1
                    print 'result: correct'
                else:
                    count_incorrect = count_incorrect + 1
                    print 'result: incorrect'
            else:
                # if np.argmax(prediction) == 1:
                if prediction[0] >= 0.5:
                    count_correct = count_correct + 1
                    print 'result: correct'
                else:
                    count_incorrect = count_incorrect + 1
                    print 'result: incorrect'
            idx = idx + 1

        for JPG_file in test_JPG_files:
            file_name = TEST_IMAGE_DIRECTORY + JPG_file
            print '#######################################################################'

            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            input_image = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

            network_output = tf.to_float(hypothesis_hand)

            x_test_data[0] = input_image
            x_test_data = x_test_data - PSEUDO_MEAN_PIXEL_VALUE
            # print x_test_data
            prediction = sess.run([network_output], feed_dict={X: x_test_data})

            print 'test_idx = ', str(idx)
            print 'input filename: ', file_name
            print 'prediction = ', prediction
            # print 'type = ', np.argmax(prediction)

            match = re.search("non_hand", jpg_file)
            if match:
                # if np.argmax(prediction) == 0:
                if prediction[0] < 0.5:
                    count_correct = count_correct + 1
                    print 'result: correct'
                else:
                    count_incorrect = count_incorrect + 1
                    print 'result: incorrect'
            else:
                # if np.argmax(prediction) == 1:
                if prediction[0] >= 0.5:
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


##############################################################################################

