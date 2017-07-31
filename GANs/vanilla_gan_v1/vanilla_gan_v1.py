import tensorflow as tf
import numpy as np

BATCH_SIZE = 30
NOISE_VECTOR_WIDTH = 10
NOISE_VECTOR_HEIGHT = 10
NOISE_VECTOR_DEPTH = 16

class SimpleGenerator:
    def __init__(self):
        #self.z = np.random.normal(size=1000)

        # placeholder
        self.X = tf.placeholder(tf.float32, [BATCH_SIZE, NOISE_VECTOR_WIDTH * NOISE_VECTOR_HEIGHT * NOISE_VECTOR_DEPTH])
        self.X = tf.reshape(self.X, [-1, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT, NOISE_VECTOR_DEPTH])

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
        self.TRANS_CONV_1 = tf.nn.relu(tf.nn.conv2d_transpose(self.X, self.TRANSPOSED_CONV_W1,
                                                              output_shape=[BATCH_SIZE, 20, 20, 8]) + self.BIAS_1)
        self.TRANS_CONV_2 = tf.nn.relu(tf.nn.conv2d_transpose(self.TRANS_CONV_1, self.TRANSPOSED_CONV_W2,
                                                              output_shape=[BATCH_SIZE, 40, 40, 4]) + self.BIAS_2)
        self.TRANS_CONV_3 = tf.nn.relu(tf.nn.conv2d_transpose(self.TRANS_CONV_2, self.TRANSPOSED_CONV_W3,
                                                              output_shape=[BATCH_SIZE, 80, 80, 1]) + self.BIAS_3)

        self.hypothesis = tf.sigmoid(self.TRANS_CONV_3)

class SimpleDiscriminator:
    def __init__(self):
        # placeholder
        self.X = tf.placeholder(tf.float32, [BATCH_SIZE, NOISE_VECTOR_WIDTH * NOISE_VECTOR_HEIGHT * NOISE_VECTOR_DEPTH])

if __name__ == '__main__':
    print 'ok'
    generator = SimpleGenerator()
