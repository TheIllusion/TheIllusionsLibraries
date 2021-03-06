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

TRAINED_MODEL_NAME = 'the_simplest_hand_classifier_v2_0.ckpt'

# Macbook Pro
ROOT_DIRECTORY = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/'

# Macbook Pro
TRAINING_LIST_FILE_NAME = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/shuffle_training_list.txt'
SOURCE_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/training_data/'
TEST_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/palm_data/hand_classifier/test_sets/test_set/'

PSEUDO_MEAN_PIXEL_VALUE = 100

LEARNING_RATE = 0.00001

if IS_TRAINING:
    BATCH_SIZE = 5
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
input_buff = np.empty(shape=(image_buffer_size, 56, 56))
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

        input_img = cv2.imread(training_file_name, cv2.IMREAD_GRAYSCALE)
        # answer_img = cv2.imread(answer_file_name, cv2.IMREAD_GRAYSCALE)

        if (type(input_img) is not np.ndarray):
            lineIdx = lineIdx + 1
            if lineIdx >= max_training_index:
                lineIdx = 0

            print 'skip this jpg file. continue.'
            continue

        input_buff[current_buff_index] = cv2.resize(input_img, (56, 56), interpolation=cv2.INTER_LINEAR)
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

##############################################################################################
#Lotto for common
X = tf.placeholder(tf.float32, [BATCH_SIZE, 56, 56])
Y = tf.placeholder(tf.float32, [BATCH_SIZE, 1])

CONV_W1_COMMON = tf.get_variable("CNN_W1_COMMON", shape=[11, 11, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
CONV_W2_COMMON = tf.get_variable("CNN_W2_COMMON", shape=[5, 5, 32, 64],
                               initializer=tf.contrib.layers.xavier_initializer())
CONV_W3_COMMON = tf.get_variable("CNN_W3_COMMON", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())

FC_W1 = tf.get_variable("FC_W1_COMMON", shape=[7 * 7 * 128, 1000], initializer=tf.contrib.layers.xavier_initializer())
FC_W2 = tf.get_variable("FC_W2_COMMON", shape=[1000, 1000], initializer=tf.contrib.layers.xavier_initializer())
FC_W3 = tf.get_variable("FC_W3_COMMON", shape=[1000, 45], initializer=tf.contrib.layers.xavier_initializer())

BIAS_CONV_W1_COMMON = tf.Variable(tf.zeros([1, 32]), name="ConvBiasCOMMON1")
BIAS_CONV_W2_COMMON = tf.Variable(tf.zeros([1, 64]), name="ConvBiasCOMMON2")
BIAS_CONV_W3_COMMON = tf.Variable(tf.zeros([1, 128]), name="ConvBiasCOMMON3")

BIAS_FC_W1 = tf.Variable(tf.zeros([1, 1000]), name="FCBiasCOMMON1")
BIAS_FC_W2 = tf.Variable(tf.zeros([1, 1000]), name="FCBiasCOMMON2")
BIAS_FC_W3 = tf.Variable(tf.zeros([1, 45]), name="FCBiasCOMMON3")

X_IN = tf.reshape(X, [-1, 56, 56, 1])

# 56->28
conv1 = tf.nn.relu(tf.nn.conv2d(X_IN, CONV_W1_COMMON, strides=[1, 2, 2, 1], padding='SAME') + BIAS_CONV_W1_COMMON)
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, CONV_W2_COMMON, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W2_COMMON)
# 28->14
conv2_pool = max_pool_2x2(conv2)
conv3 = tf.nn.relu(tf.nn.conv2d(conv2_pool, CONV_W3_COMMON, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W3_COMMON)
# 14->7
conv3_pool = max_pool_2x2(conv3)

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

FC_IN = tf.reshape(conv3_pool, [-1, 7 * 7 * 128])
FC_IN_DROPOUT = tf.nn.dropout(FC_IN, keep_prob)

FC1 = tf.nn.relu(tf.matmul(FC_IN_DROPOUT, FC_W1) + BIAS_FC_W1)
FC1_DROPOUT = tf.nn.dropout(FC1, keep_prob)

FC2 = tf.nn.relu(tf.matmul(FC1, FC_W2) + BIAS_FC_W2)
FC2_DROPOUT = tf.nn.dropout(FC2, keep_prob)

##############################################################################################
#Lotto for indivisuals

X_BIRTH = tf.placeholder(tf.float32, [BATCH_SIZE, 7, 7])
X_BIRTH_IN = tf.reshape(X_BIRTH, [-1, 7, 7, 1])

CONV_W1_INDIV = tf.get_variable("CNN_W1_INDIV", shape=[3, 3, 129, 129], initializer=tf.contrib.layers.xavier_initializer())
CONV_W2_INDIV = tf.get_variable("CNN_W2_INDIV", shape=[3, 3, 129, 129], initializer=tf.contrib.layers.xavier_initializer())
CONV_W3_INDIV = tf.get_variable("CNN_W3_INDIV", shape=[3, 3, 129, 5], initializer=tf.contrib.layers.xavier_initializer())

BIAS_CONV_W1_INDIV = tf.Variable(tf.zeros([1, 129]), name="ConvBiasINDIV1")
BIAS_CONV_W2_INDIV = tf.Variable(tf.zeros([1, 129]), name="ConvBiasINDIV2")
BIAS_CONV_W3_INDIV = tf.Variable(tf.zeros([1, 5]), name="ConvBiasINDIV3")

#concatenate filters (X_BIRTH_IN and conv3_pool)
CNN_INDIV1 = tf.nn.relu(tf.nn.conv2d(tf.concat(3, [X_BIRTH_IN, conv3_pool]), CONV_W1_INDIV, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W1_INDIV)

CNN_INDIV2 = tf.nn.relu(tf.nn.conv2d(CNN_INDIV1, CONV_W2_INDIV, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W2_INDIV)
CNN_INDIV3 = tf.nn.sigmoid(tf.nn.conv2d(CNN_INDIV2, CONV_W3_INDIV, strides=[1, 1, 1, 1], padding='SAME') + BIAS_CONV_W3_INDIV)

##############################################################################################
#hypothesis_hand = tf.nn.sigmoid(tf.matmul(FC2_DROPOUT, FC_W3) + BIAS_FC_W3)
hypothesis_hand = tf.nn.softmax(tf.matmul(FC2_DROPOUT, FC_W3) + BIAS_FC_W3)
hypothesis_hand_indiv = CNN_INDIV3

##############################################################################################
if IS_TRAINING:
    regularization_term = 0.0005 * (tf.nn.l2_loss(CONV_W1_COMMON)
                                      + tf.nn.l2_loss(CONV_W2_COMMON)
                                      + tf.nn.l2_loss(CONV_W3_COMMON)
                                      + tf.nn.l2_loss(CONV_W1_INDIV)
                                      + tf.nn.l2_loss(CONV_W2_INDIV)
                                      + tf.nn.l2_loss(CONV_W3_INDIV)
                                      + tf.nn.l2_loss(FC_W1)
                                      + tf.nn.l2_loss(FC_W2)
                                      + tf.nn.l2_loss(FC_W3))

    cost = (-tf.reduce_mean(Y * tf.log(tf.clip_by_value(hypothesis_hand, 1e-30, 1.0))
                                 + (1 - Y) * tf.log(tf.clip_by_value(1 - hypothesis_hand, 1e-30, 1.0)))
                 + regularization_term)

    # learning rate
    learning_rate_hand = tf.Variable(LEARNING_RATE)

    optimizer_hand = tf.train.AdamOptimizer(learning_rate_hand)
    train_hand = optimizer_hand.minimize(cost)

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
            x_data = np.empty(shape=(BATCH_SIZE, 56, 56))
            y_data = np.empty(shape=(BATCH_SIZE, 1))
            x_birth = np.empty(shape=(BATCH_SIZE, 7, 7))

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

                # Random generation for birth data (for temporary)
                x_birth[batchIdx] = np.random.rand(7, 7)

            sess.run(train_hand, feed_dict={X: x_data, Y: y_data, X_BIRTH: x_birth, keep_prob: 0.7})
            # sess.run(train_hand, feed_dict={X: x_data, Y: y_data, dropout_rate_hand: 0.7})
            if step % 50 == 0:
                print 'step = ', step
                print 'regularization cost = ', sess.run(regularization_term)
                print 'total cost = ', sess.run(cost, feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
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

        x_test_data = np.empty(shape=(1, 56, 56))

        for jpg_file in test_jpg_files:
            file_name = TEST_IMAGE_DIRECTORY + jpg_file
            print '#######################################################################'

            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            input_image = cv2.resize(img, (56, 56), interpolation=cv2.INTER_LINEAR)

            network_output = tf.to_float(hypothesis_hand)
            print 'after mean subtraction. PSEUDO_MEAN = ', PSEUDO_MEAN_PIXEL_VALUE
            x_test_data[0] = input_image
            x_test_data = x_test_data - PSEUDO_MEAN_PIXEL_VALUE
            # print x_test_data
            # print input_image
            prediction = sess.run([network_output], feed_dict={X: x_test_data, keep_prob: 1.0})

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

            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            input_image = cv2.resize(img, (56, 56), interpolation=cv2.INTER_LINEAR)

            network_output = tf.to_float(hypothesis_hand)

            x_test_data[0] = input_image
            x_test_data = x_test_data - PSEUDO_MEAN_PIXEL_VALUE
            # print x_test_data
            prediction = sess.run([network_output], feed_dict={X: x_test_data, keep_prob: 1.0})

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

