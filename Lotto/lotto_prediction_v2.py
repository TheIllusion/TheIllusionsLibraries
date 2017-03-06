import tensorflow as tf
import numpy as np

IS_TRAINING = True

TRAINED_MODEL_NAME = 'lotto_prediction_v2_1000.ckpt'

# Macbook Pro
ROOT_DIRECTORY = '/Users/Illusion/PycharmProjects/TheIllusionsLibraries/Lotto/'

# Macbook Pro
TRAINING_LIST_FILE_NAME = ROOT_DIRECTORY + 'lotto_data.txt'

LEARNING_RATE = 0.00001

if IS_TRAINING:
    BATCH_SIZE = 1
else:
    BATCH_SIZE = 1

training_list_file = open(TRAINING_LIST_FILE_NAME)
training_list_temp = training_list_file.readlines()

max_training_index = len(training_list_temp)

#split numbers from each line of string
training_list_index = 0
max_test_size = 5

training_list = []
for line_str in training_list_temp:
    training_list.append(line_str.split())

##############################################################################################

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

##############################################################################################

#Lotto for common
X = tf.placeholder(tf.float32, [BATCH_SIZE, 56, 56])
Y = tf.placeholder(tf.float32, [BATCH_SIZE, 45])

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

hypothesis = tf.nn.sigmoid(tf.matmul(FC2_DROPOUT, FC_W3) + BIAS_FC_W3)
#hypothesis = tf.nn.softmax(tf.matmul(FC2_DROPOUT, FC_W3) + BIAS_FC_W3)

##############################################################################################
if IS_TRAINING:
    regularization_term = 0.0003 * (tf.nn.l2_loss(CONV_W1_COMMON)
                                      + tf.nn.l2_loss(CONV_W2_COMMON)
                                      + tf.nn.l2_loss(CONV_W3_COMMON)
                                      + tf.nn.l2_loss(FC_W1)
                                      + tf.nn.l2_loss(FC_W2)
                                      + tf.nn.l2_loss(FC_W3))

    cost = (-tf.reduce_mean(Y * tf.log(tf.clip_by_value(hypothesis, 1e-30, 1.0))
                                 + (1 - Y) * tf.log(tf.clip_by_value(1 - hypothesis, 1e-30, 1.0)))
                 + regularization_term)

    # learning rate
    learning_rate_hand = tf.Variable(LEARNING_RATE)

    optimizer_hand = tf.train.AdamOptimizer(learning_rate_hand)
    train_lotto = optimizer_hand.minimize(cost)

    # dropout rate
    dropout_rate_hand = tf.placeholder("float")

    init_lotto = tf.initialize_all_variables()

# tf.global_variables_initializer()

##############################################################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    if IS_TRAINING:
        sess.run(init_lotto)

        # train the model
        for step in xrange(500001):
            x_data = np.zeros(shape=(BATCH_SIZE, 56, 56))
            y_data = np.zeros(shape=(BATCH_SIZE, 45))

            #Loop through batch
            for batchIdx in xrange(BATCH_SIZE):
                #Loop through lines of X
                for line_index in xrange(56):
                    # Process each line of input X
                    for i in xrange(6):
                        #Put 1 at the right locations
                        x_data[batchIdx][line_index][int(training_list[training_list_index][i])] = 1

                    training_list_index = training_list_index + 1

                    if training_list_index >= len(training_list):
                        training_list_index = 0

                #print 'Line index of training list = ', training_list_index

                #Process Y
                for i in xrange(6):
                    #Lotto answer numbers have range from 1 to 45
                    #Convert the range to 0 to 44
                    #Put the probability 1 to each location
                    y_data[batchIdx][int(training_list[training_list_index][i]) - 1] = 1.0

            sess.run(train_lotto, feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})

            if step % 50 == 0:
                print 'step = ', step
                print 'regularization cost = ', sess.run(regularization_term)
                print 'total cost = ', sess.run(cost, feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
                print 'hypothesis = ', sess.run(hypothesis, feed_dict={X: x_data, Y: y_data, keep_prob: 0.7})
                print 'answer = ', training_list[training_list_index]
                print '-------------------------------'

            if step % 1000 == 0:
                saver = tf.train.Saver(tf.trainable_variables())
                file_name = ROOT_DIRECTORY + "lotto_prediction_v2_" + str(step) + ".ckpt"
                save_path = saver.save(sess, file_name)
                print("Model saved in file: %s" % save_path)

    # Feed-forward Test
    else:
        # Load the pre-trained model from file
        saver = tf.train.Saver()
        saver.restore(sess, ROOT_DIRECTORY + TRAINED_MODEL_NAME)

        # sess.run(init_hand)
        # reader = tf.train.NewCheckpointReader(ROOT_DIRECTORY + 'models/' + TRAINED_MODEL_NAME)

        print("Model restored.")

    ##############################################################################################
    # Test the trained model
    # Read the test data
    if not IS_TRAINING:

        x_test_data = np.empty(shape=(1, 56, 56))

        training_list_index = 0

        #Loop through each test
        for test_idx in xrange(max_test_size):
            # Loop through lines of X
            for line_index in xrange(56):
                # Process each line of input X
                for i in xrange(6):
                    # Put 1 at the right locations
                    x_test_data[0][line_index][int(training_list[training_list_index][i])] = 1

                training_list_index = training_list_index + 1

                if training_list_index >= len(training_list):
                    training_list_index = 0

            #sess.run(train_lotto, feed_dict={X: x_data, keep_prob: 1.0})
            print 'hypothesis = ', sess.run(hypothesis, feed_dict={X: x_test_data, keep_prob: 1.0})
            print 'answer = ', training_list[training_list_index]
            print '==================================================================='

        print 'End'


##############################################################################################

