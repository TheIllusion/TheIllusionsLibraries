# using MNIST data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import os
import time
import random
import cv2

#MNIST_DATA_SAVING_DIR = "/tmp/tensorflow/mnist/input_data/"
MNIST_DATA_SAVING_DIR = "./mnist/input_data/"

# output image save directory
# Macbook Pro
OUTPUT_IMAGE_SAVE_DIRECTORY = "/Users/Illusion/Downloads/vanilla_gan_generated/"
# i7-2600k (Ubuntu)
#OUTPUT_IMAGE_SAVE_DIRECTORY = "/media/illusion/ML_Linux/temp/vanilla_gan_v1_gen_images/"

# Import data [784] = [28x28]
mnist = input_data.read_data_sets(MNIST_DATA_SAVING_DIR, one_hot=True)

IS_TRAINING = True

TOTAL_ITERATION = 500000

BATCH_SIZE = 30

# The size of the latent vector space seems to be really important
# if the size is too big, the output image shows sparse distributions
NOISE_VECTOR_WIDTH = 1
NOISE_VECTOR_HEIGHT = 1
NOISE_VECTOR_DEPTH = 1

INPUT_IMAGE_WIDTH = 28
INPUT_IMAGE_HEIGHT = 28

# learning rate
initial_learning_rate_disc = tf.Variable(0.00001)
initial_learning_rate_gen = tf.Variable(0.0001)

##############################################################################################

class SimpleGenerator:
    def __init__(self, sess, discriminator):
        with tf.variable_scope("generator") as scope:

            self.sess = sess

            #self.z = np.random.normal(size=1000)

            # placeholder
            self.gen_X = tf.placeholder(tf.float32, [BATCH_SIZE, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT, NOISE_VECTOR_DEPTH])

            # weights
            self.gen_TRANSPOSED_CONV_W1 = tf.get_variable("GEN_TRANSPOSED_CONV_W1", shape=[2, 2, 64, NOISE_VECTOR_DEPTH],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W2 = tf.get_variable("GEN_TRANSPOSED_CONV_W2", shape=[2, 2, 128, 64],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W3 = tf.get_variable("GEN_TRANSPOSED_CONV_W3", shape=[2, 2, 128, 128],
                                                          initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W4 = tf.get_variable("GEN_TRANSPOSED_CONV_W4", shape=[2, 2, 128, 128],
                                                          initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W5 = tf.get_variable("GEN_TRANSPOSED_CONV_W5", shape=[2, 2, 32, 128],
                                                          initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W6 = tf.get_variable("GEN_TRANSPOSED_CONV_W6", shape=[2, 2, 1, 32],
                                                      initializer=tf.contrib.layers.xavier_initializer())

            # biases
            self.gen_BIAS_1 = tf.Variable(tf.zeros([1, 64]), name="GEN_BIAS_1")
            self.gen_BIAS_2 = tf.Variable(tf.zeros([1, 128]), name="GEN_BIAS_2")
            self.gen_BIAS_3 = tf.Variable(tf.zeros([1, 128]), name="GEN_BIAS_3")
            self.gen_BIAS_4 = tf.Variable(tf.zeros([1, 128]), name="GEN_BIAS_4")
            self.gen_BIAS_5 = tf.Variable(tf.zeros([1, 32]), name="GEN_BIAS_5")
            self.gen_BIAS_6 = tf.Variable(tf.zeros([1, 1]), name="GEN_BIAS_6")

            #tf.global_variables_initializer().run()

            self.var_list_gen = [self.gen_TRANSPOSED_CONV_W1, self.gen_TRANSPOSED_CONV_W2, self.gen_TRANSPOSED_CONV_W3,
                                 self.gen_TRANSPOSED_CONV_W4, self.gen_TRANSPOSED_CONV_W5, self.gen_TRANSPOSED_CONV_W6,
                                 self.gen_BIAS_1, self.gen_BIAS_2, self.gen_BIAS_3,
                                 self.gen_BIAS_4, self.gen_BIAS_5, self.gen_BIAS_6]

    def forward(self, x):
        # graph
        gen_TRANS_CONV_1 = tf.nn.relu(tf.nn.conv2d_transpose(x,
                                                              self.gen_TRANSPOSED_CONV_W1,
                                                              output_shape=[BATCH_SIZE, 2, 2, 64],
                                                              strides=[1, 2, 2, 1],
                                                              padding="SAME") + self.gen_BIAS_1)

        gen_TRANS_CONV_2 = tf.nn.relu(tf.nn.conv2d_transpose(gen_TRANS_CONV_1,
                                                             self.gen_TRANSPOSED_CONV_W2,
                                                             output_shape=[BATCH_SIZE, 4, 4, 128],
                                                             strides=[1, 2, 2, 1],
                                                             padding="SAME") + self.gen_BIAS_2)

        gen_TRANS_CONV_3 = tf.nn.relu(tf.nn.conv2d_transpose(gen_TRANS_CONV_2,
                                                             self.gen_TRANSPOSED_CONV_W3,
                                                             output_shape=[BATCH_SIZE, 7, 7, 128],
                                                             strides=[1, 2, 2, 1],
                                                             padding="SAME") + self.gen_BIAS_3)

        gen_TRANS_CONV_4 = tf.nn.relu(tf.nn.conv2d_transpose(gen_TRANS_CONV_3,
                                                             self.gen_TRANSPOSED_CONV_W4,
                                                             output_shape=[BATCH_SIZE, 13, 13, 128],
                                                             strides=[1, 2, 2, 1],
                                                             padding="SAME") + self.gen_BIAS_4)

        gen_TRANS_CONV_5 = tf.nn.relu(tf.nn.conv2d_transpose(gen_TRANS_CONV_4,
                                                              self.gen_TRANSPOSED_CONV_W5,
                                                              output_shape=[BATCH_SIZE, 14, 14, 32],
                                                              strides=[1, 1, 1, 1],
                                                              padding="VALID") + self.gen_BIAS_5)

        gen_TRANS_CONV_6 = tf.nn.conv2d_transpose(gen_TRANS_CONV_5,
                                                   self.gen_TRANSPOSED_CONV_W6,
                                                   output_shape=[BATCH_SIZE, 28, 28, 1],
                                                   strides=[1, 2, 2, 1],
                                                   padding="VALID") + self.gen_BIAS_6

        gen_hypothesis = tf.sigmoid(gen_TRANS_CONV_6)

        gen_hypothesis = tf.reshape(gen_hypothesis, [-1, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT])

        return gen_hypothesis

    def generate_fake_imgs(self):
        # Random noise vector

        noise_z = np.random.uniform(-1, 1, [BATCH_SIZE, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT, NOISE_VECTOR_DEPTH]).astype(np.float32)

        gen_hypothesis = self.forward(self.gen_X)
        network_output = self.sess.run(gen_hypothesis, feed_dict={self.gen_X: noise_z})

        # scale to 0~255
        fake_imgs = network_output * 255

        self.fake_imgs = fake_imgs

        return fake_imgs

class SimpleDiscriminator:
    def __init__(self, sess):
        with tf.variable_scope("discriminator") as scope:

            #self.real_img_buff = np.empty(shape=(BATCH_SIZE, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 1))
            self.real_img_buff = np.empty(shape=(BATCH_SIZE, 784))

            self.sess = sess

            # placeholder
            #self.disc_X = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_DEPTH])
            self.disc_X = tf.placeholder(tf.float32,
                                         [BATCH_SIZE, INPUT_IMAGE_WIDTH*INPUT_IMAGE_HEIGHT])

            # weights for convolutional layers
            self.disc_CNN_W1 = tf.get_variable("DISC_CNN_W1", shape=[5, 5, 1, 16], initializer=tf.contrib.layers.xavier_initializer())
            self.disc_CNN_W2 = tf.get_variable("DISC_CNN_W2", shape=[3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
            self.disc_CNN_W3 = tf.get_variable("DISC_CNN_W3", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())

            self.disc_FC_W1 = tf.get_variable("DISC_FC_W1", shape=[12544, 1000],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.disc_FC_W2 = tf.get_variable("DISC_FC_W2", shape=[1000, 1000],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.disc_FC_W3 = tf.get_variable("DISC_FC_W3", shape=[1000, 1],
                                       initializer=tf.contrib.layers.xavier_initializer())

            # biases
            self.disc_BIAS_CONV_1 = tf.Variable(tf.zeros([1, 16]), name="DISC_ConvBias1")
            self.disc_BIAS_CONV_2 = tf.Variable(tf.zeros([1, 32]), name="DISC_ConvBias2")
            self.disc_BIAS_CONV_3 = tf.Variable(tf.zeros([1, 64]), name="DISC_ConvBias3")
            self.disc_BIAS_FC_W1 = tf.Variable(tf.zeros([1, 1000]), name="DISC_FCBias1")
            self.disc_BIAS_FC_W2 = tf.Variable(tf.zeros([1, 1000]), name="DISC_FCBias2")
            self.disc_BIAS_FC_W3 = tf.Variable(tf.zeros([1, 1]), name="DISC_FCBias3")

            self.var_list_disc = [self.disc_CNN_W1, self.disc_CNN_W2, self.disc_CNN_W3,
                                  self.disc_FC_W1, self.disc_FC_W2, self.disc_FC_W3,
                                  self.disc_BIAS_CONV_1, self.disc_BIAS_CONV_2, self.disc_BIAS_CONV_3,
                                  self.disc_BIAS_FC_W1, self.disc_BIAS_FC_W2, self.disc_BIAS_FC_W3]

    def feed_images(self, fake_imgs):
        # real images are already fed
        self.fake_imgs = fake_imgs

    def forward(self, x):

        x = tf.reshape(x, [-1, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 1])

        # graph
        # 28x28
        disc_CNN1 = tf.nn.relu(tf.nn.conv2d(x, self.disc_CNN_W1, strides=[1, 1, 1, 1], padding='SAME') + self.disc_BIAS_CONV_1)
        disc_CNN2 = tf.nn.relu(tf.nn.conv2d(disc_CNN1, self.disc_CNN_W2, strides=[1, 1, 1, 1],
                                                 padding='SAME') + self.disc_BIAS_CONV_2)
        disc_CNN3 = tf.nn.relu(tf.nn.conv2d(disc_CNN2, self.disc_CNN_W3, strides=[1, 2, 2, 1],
                                                 padding='SAME') + self.disc_BIAS_CONV_3)

        # 14x14
        disc_FC_IN = tf.reshape(disc_CNN3, [-1, 14 * 14 * 64])
        disc_FC1 = tf.nn.relu(tf.matmul(disc_FC_IN, self.disc_FC_W1) + self.disc_BIAS_FC_W1)
        disc_FC2 = tf.nn.relu(tf.matmul(disc_FC1, self.disc_FC_W2) + self.disc_BIAS_FC_W2)

        disc_hypothesis = tf.nn.sigmoid(tf.matmul(disc_FC2, self.disc_FC_W3) + self.disc_BIAS_FC_W3)

        return disc_hypothesis

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_IMAGE_SAVE_DIRECTORY):
        os.mkdir(OUTPUT_IMAGE_SAVE_DIRECTORY)

    # Create a Tensorflow session
    with tf.Session() as sess:

        # create a discriminator
        discriminator = SimpleDiscriminator(sess)

        # create a generator
        generator = SimpleGenerator(sess, discriminator)

        # read real imgs
        image_buff_read_index = 0

        # loss functions for discriminator
        disc_real = discriminator.forward(discriminator.disc_X)
        disc_fake = discriminator.forward(generator.forward(generator.gen_X))
        disc_total_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

        optimizer_discriminator = tf.train.AdamOptimizer(initial_learning_rate_disc)
        optimizer_generator = tf.train.AdamOptimizer(initial_learning_rate_gen)

        # loss function for generator
        gen_loss = -tf.reduce_mean(tf.log(disc_fake))

        # optimize the parameters of the discriminator only
        train_disc = optimizer_discriminator.minimize(disc_total_loss, var_list=discriminator.var_list_disc)

        # optimize the parameters of the generator only
        train_gen = optimizer_generator.minimize(gen_loss, var_list=generator.var_list_gen)

        #
        # init = tf.initialize_all_variables()
        # sess.run(init)
        tf.global_variables_initializer().run()

        for iter in range(TOTAL_ITERATION):

            # print 'iter: ', str(iter)

            for i in range(BATCH_SIZE):

                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

                # should I copy this with for loops?
                np.copyto(discriminator.real_img_buff[i], batch_xs[image_buff_read_index])

            # noise vector z
            noise_z = np.random.uniform(-1, 1, [BATCH_SIZE, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT,
                                                NOISE_VECTOR_DEPTH]).astype(np.float32)

            # train the discriminator
            _, disc_loss_current = sess.run([train_disc, disc_total_loss],
                                            feed_dict={discriminator.disc_X: discriminator.real_img_buff, generator.gen_X: noise_z})

            # train the generator
            _, gen_loss_current_1 = sess.run([train_gen, gen_loss], feed_dict={generator.gen_X: noise_z})

            # train the generator (2nd time)
            #_, gen_loss_current_2 = sess.run([train_gen, gen_loss], feed_dict={generator.gen_X: noise_z})

            # check the loss every 10-iter
            if iter % 100 == 0:
                print '=============================================='
                print 'iter = ', str(iter)
                print 'discriminator loss: ', str(disc_loss_current)
                print 'generator loss(1): ', str(gen_loss_current_1)
                #print 'generator loss(2): ', str(gen_loss_current_2)
                #print 'generator loss(3): ', str(gen_loss_current_3)
                #print 'generator loss(4): ', str(gen_loss_current_4)
                #print 'generator loss(5): ', str(gen_loss_current_5)
                print '=============================================='

            # generate fake iamges every 100-iter to check the quality of output images
            if iter % 1000 == 0:
                fake_imgs = generator.generate_fake_imgs()
                for j in range(len(fake_imgs)):
                    # save images to jpg files
                    filename = "iter_" + str(iter) + "_generated_" + str(j) + ".jpg"
                    cv2.imwrite(OUTPUT_IMAGE_SAVE_DIRECTORY + filename, fake_imgs[j])

    exit_notification = True

    print 'ok'
