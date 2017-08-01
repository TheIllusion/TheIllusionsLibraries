import tensorflow as tf
import numpy as np
import os
import glob
import random

BATCH_SIZE = 30
NOISE_VECTOR_WIDTH = 10
NOISE_VECTOR_HEIGHT = 10
NOISE_VECTOR_DEPTH = 16

INPUT_IMAGE_DIRECTORY_PATH = "/Users/Illusion/Documents/Data/face_data/20_female/"

class SimpleGenerator:
    def __init__(self):
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

INPUT_IMAGE_WIDTH = 80
INPUT_IMAGE_HEIGHT = 80
INPUT_IMAGE_DEPTH = 3

class SimpleDiscriminator:
    def __init__(self):
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

    # create a generator
    generator = SimpleGenerator()

    # create a discriminator
    discriminator = SimpleDiscriminator()

    # load the filelist
    os.chdir(INPUT_IMAGE_DIRECTORY_PATH)
    jpg_files = glob.glob('*.jpg')
    random.shuffle(jpg_files)

    print 'ok'

