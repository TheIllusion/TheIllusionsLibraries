import tensorflow as tf
import numpy as np
import os, re
import time
import glob
import random
import cv2
import threading

IS_TRAINING = True

BATCH_SIZE = 30
NOISE_VECTOR_WIDTH = 10
NOISE_VECTOR_HEIGHT = 10
NOISE_VECTOR_DEPTH = 16

INPUT_IMAGE_WIDTH = 80
INPUT_IMAGE_HEIGHT = 80
INPUT_IMAGE_DEPTH = 3

PSEUDO_MEAN_PIXEL_VALUE = 100

# Macbook Pro
#INPUT_IMAGE_DIRECTORY_PATH = "/Users/Illusion/Documents/Data/face_data/20_female/"

# Macbook 12
#INPUT_IMAGE_DIRECTORY_PATH = "/Users/Illusion/Documents/Caricature/face_refined_1/"

# i7-2600k (Ubuntu)
INPUT_IMAGE_DIRECTORY_PATH = "/media/illusion/ML_Linux/Data/KCeleb-v1/kim.yuna/"

##############################################################################################
# Image Buffer Management

# image buffers
image_buffer_size = 200
input_buff = np.empty(shape=(image_buffer_size, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3))
#answer_buff = np.empty(shape=(image_buffer_size))

buff_status = []
for i in range(image_buffer_size):
    buff_status.append('empty')

current_buff_index = 0
lineIdx = 0

# load the filelist
os.chdir(INPUT_IMAGE_DIRECTORY_PATH)
jpg_files = glob.glob('*.jpg')
random.shuffle(jpg_files)

#training_list_file = open(TRAINING_LIST_FILE_NAME)
#training_list = training_list_file.readlines()
max_training_index = len(jpg_files)

exit_notification = False

def image_buffer_loader():
    global current_buff_index
    global lineIdx

    print 'image_buffer_loader'

    while True:
        filename_ = jpg_files[lineIdx]

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

        training_file_name = filename
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

        input_buff[current_buff_index] = cv2.resize(input_img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        # classifying_string = filename_[end_index + 1]

        '''
        match = re.search("right_hand", filename_)
        if match:
            classifying_string = "0"
        else:
            classifying_string = "1"

        answer_buff[current_buff_index] = float(classifying_string)
        '''

        buff_status[current_buff_index] = 'filled'

        # pseudo_mean_value = np.mean(input_buff[current_buff_index])
        input_buff[current_buff_index] = input_buff[current_buff_index] - PSEUDO_MEAN_PIXEL_VALUE

        if lineIdx % 10 == 0:
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
##############################################################################################

class SimpleGenerator:
    def __init__(self, sess):

        self.sess = sess
        #self.z = np.random.normal(size=1000)

        # placeholder
        self.X = tf.placeholder(tf.float32, [BATCH_SIZE, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT, NOISE_VECTOR_DEPTH])
        #self.X = tf.reshape(self.X, [-1, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT, NOISE_VECTOR_DEPTH])

        # weights
        self.TRANSPOSED_CONV_W1 = tf.get_variable("TRANSPOSED_CONV_W1", shape=[2, 2, 8, 16],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.TRANSPOSED_CONV_W2 = tf.get_variable("TRANSPOSED_CONV_W2", shape=[2, 2, 4, 8],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.TRANSPOSED_CONV_W3 = tf.get_variable("TRANSPOSED_CONV_W3", shape=[2, 2, 1, 4],
                                                  initializer=tf.contrib.layers.xavier_initializer())

        # biases
        self.BIAS_1 = tf.Variable(tf.zeros([1, 8]), name="BIAS_1")
        self.BIAS_2 = tf.Variable(tf.zeros([1, 4]), name="BIAS_2")
        self.BIAS_3 = tf.Variable(tf.zeros([1, 1]), name="BIAS_3")

        # graph
        self.TRANS_CONV_1 = tf.nn.relu(tf.nn.conv2d_transpose(self.X,
                                                              self.TRANSPOSED_CONV_W1,
                                                              output_shape=[BATCH_SIZE, 20, 20, 8],
                                                              strides=[1,2,2,1],
                                                              padding="SAME") + self.BIAS_1)

        self.TRANS_CONV_2 = tf.nn.relu(tf.nn.conv2d_transpose(self.TRANS_CONV_1,
                                                              self.TRANSPOSED_CONV_W2,
                                                              output_shape=[BATCH_SIZE, 40, 40, 4],
                                                              strides=[1, 2, 2, 1],
                                                              padding="SAME") + self.BIAS_2)

        self.TRANS_CONV_3 = tf.nn.relu(tf.nn.conv2d_transpose(self.TRANS_CONV_2,
                                                              self.TRANSPOSED_CONV_W3,
                                                              output_shape=[BATCH_SIZE, 80, 80, 1],
                                                              strides=[1, 2, 2, 1],
                                                              padding="SAME") + self.BIAS_3)

        self.hypothesis = tf.sigmoid(self.TRANS_CONV_3)

    def generate_fake_imgs(self):
        # Random noise vector

        noise_z = np.random.uniform(-1, 1, [BATCH_SIZE, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT, NOISE_VECTOR_DEPTH]).astype(np.float32)

        network_output = self.sess.run(self.hypothesis, feed_dict={self.X: noise_z})

        # network_output = tf.to_float(self.hypothesis)

        # scale to 0~255
        fake_imgs = network_output * 255

        return fake_imgs

class SimpleDiscriminator:
    def __init__(self, sess):

        self.sess = sess

        # placeholder
        self.X = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_DEPTH])

        # weights for convolutional layers
        self.CNN_W1 = tf.get_variable("CNN_W1", shape=[5, 5, 3, 16], initializer=tf.contrib.layers.xavier_initializer())
        self.CNN_W2 = tf.get_variable("CNN_W2", shape=[3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
        self.CNN_W3 = tf.get_variable("CNN_W3", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())

        self.FC_W1 = tf.get_variable("FC_W1", shape=[10 * 10 * 64, 1000],
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.FC_W2 = tf.get_variable("FC_W2", shape=[1000, 1000],
                                   initializer=tf.contrib.layers.xavier_initializer())
        self.FC_W3 = tf.get_variable("FC_W3", shape=[1000, 1],
                                   initializer=tf.contrib.layers.xavier_initializer())

        # biases
        self.BIAS_CONV_1 = tf.Variable(tf.zeros([1, 16]), name="ConvBias1")
        self.BIAS_CONV_2 = tf.Variable(tf.zeros([1, 32]), name="ConvBias2")
        self.BIAS_CONV_3 = tf.Variable(tf.zeros([1, 64]), name="ConvBias3")
        self.BIAS_FC_W1 = tf.Variable(tf.zeros([1, 1000]), name="FCBias1")
        self.BIAS_FC_W2 = tf.Variable(tf.zeros([1, 1000]), name="FCBias2")
        self.BIAS_FC_W3 = tf.Variable(tf.zeros([1, 1]), name="FCBias3")

        # graph
        self.CNN1 = tf.nn.relu(tf.nn.conv2d(self.X, self.CNN_W1, strides=[1, 2, 2, 1], padding='SAME') + self.BIAS_CONV_1)
        self.CNN2 = tf.nn.relu(tf.nn.conv2d(self.CNN1, self.CNN_W2, strides=[1, 2, 2, 1], padding='SAME') + self.BIAS_CONV_2)
        self.CNN3 = tf.nn.relu(tf.nn.conv2d(self.CNN2, self.CNN_W3, strides=[1, 2, 2, 1], padding='SAME') + self.BIAS_CONV_3)

        self.FC_IN = tf.reshape(self.CNN3, [-1, 10 * 10 * 64])
        self.FC1 = tf.nn.relu(tf.matmul(self.FC_IN, self.FC_W1) + self.BIAS_FC_W1)
        self.FC2 = tf.nn.relu(tf.matmul(self.FC1, self.FC_W2) + self.BIAS_FC_W2)

        self.hypothesis = tf.nn.sigmoid(tf.matmul(self.FC2, self.FC_W3) + self.BIAS_FC_W3)

if __name__ == '__main__':

    # Create a Tensorflow session
    with tf.Session() as sess:
        # create a generator
        generator = SimpleGenerator(sess)

        # create a discriminator
        discriminator = SimpleDiscriminator(sess)

        #
        init = tf.initialize_all_variables()
        sess.run(init)

        fake_imgs = generator.generate_fake_imgs()

    print 'ok'

