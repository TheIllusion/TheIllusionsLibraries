import tensorflow as tf

HAND_IMAGE_DIRECTORY = '/media/illusion/ML_Linux/Data/hand_detection/hands'
NON_HAND_IMAGE_DIRECTORY = '/media/illusion/ML_Linux/Data/hand_detection/non_hands'

BATCH_SIZE = 50

PSEUDO_MEAN_PIXEL_VALUE = 100

X = tf.placeholder(tf.float32, [BATCH_SIZE, 512, 512, 3])

CONV_W1_HAND = tf.get_variable("CNN_W1_HAND", shape=[5,5,3, 32], initializer=tf.contrib.layers.xavier_initializer())
CONV_W2_HAND = tf.get_variable("CNN_W1_HAND", shape=[5,5,3, 64], initializer=tf.contrib.layers.xavier_initializer())
CONV_W3_HAND = tf.get_variable("CNN_W1_HAND", shape=[5,5,3, 128], initializer=tf.contrib.layers.xavier_initializer())
CONV_W4_HAND = tf.get_variable("CNN_W1_HAND", shape=[5,5,3, 256], initializer=tf.contrib.layers.xavier_initializer())
CONV_W5_HAND = tf.get_variable("CNN_W1_HAND", shape=[5,5,3, 512], initializer=tf.contrib.layers.xavier_initializer())
CONV_W6_HAND = tf.get_variable("CNN_W1_HAND", shape=[5,5,3, 1024], initializer=tf.contrib.layers.xavier_initializer())

BIAS_CONV_W1_HAND = tf.Variable(tf.zeros([1,32]), name = "ConvBiasHand1")
BIAS_CONV_W2_HAND = tf.Variable(tf.zeros([1,64]), name = "ConvBiasHand2")
BIAS_CONV_W3_HAND = tf.Variable(tf.zeros([1,128]), name = "ConvBiasHand3")
BIAS_CONV_W4_HAND = tf.Variable(tf.zeros([1,256]), name = "ConvBiasHand4")
BIAS_CONV_W5_HAND = tf.Variable(tf.zeros([1,512]), name = "ConvBiasHand5")
BIAS_CONV_W6_HAND = tf.Variable(tf.zeros([1,1024]), name = "ConvBiasHand6")
BIAS_FC_W1 = tf.Variable(tf.zeros([1,1024]), name = "ConvBiasHand6")

X_IN = tf.reshape(X, [-1, 512, 512, 3])

CNN1 = tf.nn.relu(tf.nn.conv2d(X_IN, CONV_W1_HAND, strides=[1, 2, 2, 1], padding = 'SAME') + BIAS_CONV_W1_HAND)
CNN2 = tf.nn.relu(tf.nn.conv2d(CNN1, CONV_W2_HAND, strides=[1, 2, 2, 1], padding = 'SAME') + BIAS_CONV_W2_HAND)
CNN3 = tf.nn.relu(tf.nn.conv2d(CNN2, CONV_W3_HAND, strides=[1, 2, 2, 1], padding = 'SAME') + BIAS_CONV_W3_HAND)
CNN4 = tf.nn.relu(tf.nn.conv2d(CNN3, CONV_W4_HAND, strides=[1, 2, 2, 1], padding = 'SAME') + BIAS_CONV_W4_HAND)
CNN5 = tf.nn.relu(tf.nn.conv2d(CNN4, CONV_W5_HAND, strides=[1, 2, 2, 1], padding = 'SAME') + BIAS_CONV_W5_HAND)
CNN6 = tf.nn.relu(tf.nn.conv2d(CNN5, CONV_W6_HAND, strides=[1, 2, 2, 1], padding = 'SAME') + BIAS_CONV_W6_HAND)

FC1_IN = tf.reshape(CNN6, [-1, 64, 1])
FC1 = tf.nn.relu(tf.matmul(CNN6, FC1_IN) + BIAS_FC_W1)
FC2 = tf.nn.relu(tf.matmul(FC1, FC1_IN) + BIAS_FC_W1)