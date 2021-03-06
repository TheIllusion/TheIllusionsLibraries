import tensorflow as tf
import numpy as np
import os, re
import time
import glob
import random
import cv2
import threading

IS_TRAINING = True

TOTAL_ITERATION = 500000

BATCH_SIZE = 50
NOISE_VECTOR_WIDTH = 4
NOISE_VECTOR_HEIGHT = 4
NOISE_VECTOR_DEPTH = 3

INPUT_IMAGE_WIDTH = 32
INPUT_IMAGE_HEIGHT = 32
INPUT_IMAGE_DEPTH = 3

PIXEL_VALUE_FOR_ZERO_CENTERED = 128

# learning rate
initial_learning_rate_disc = tf.Variable(0.0002)
initial_learning_rate_gen = tf.Variable(0.0002)

# svc002
#INPUT_IMAGE_DIRECTORY_PATH = "/home1/irteamsu/users/rklee/TheIllusionsLibraries/GANs/vanilla_gan_v1/blonde_hair"
#"/home1/irteamsu/users/rklee/TheIllusionsLibraries/GANs/vanilla_gan_v1/face_imgs_svc"

# Macbook Pro
#INPUT_IMAGE_DIRECTORY_PATH = "/Users/Illusion/Documents/Data/face_data/20_female/"

# Macbook 12
#INPUT_IMAGE_DIRECTORY_PATH = "/Users/Illusion/Documents/Caricature/face_refined_1/original/"

# tbt003
INPUT_IMAGE_DIRECTORY_PATH = '/home1/irteamsu/data/rklee/hair_segmentation/seg_result_until_20170911_and_lfw/total_augmented_training_data/input_imgs/'

# i7-2600k (Ubuntu)
#INPUT_IMAGE_DIRECTORY_PATH = "/media/illusion/ML_DATA_SSD_M550/KCeleb-all-faces/"
#INPUT_IMAGE_DIRECTORY_PATH = '/media/illusion/ML_Linux/Data/hair_segmentation/original_all/original_all/'

# output image save directory
# Macbook Pro
#OUTPUT_IMAGE_SAVE_DIRECTORY = "/Users/Illusion/Downloads/vanilla_gan_generated/"
# svc002
#OUTPUT_IMAGE_SAVE_DIRECTORY = "/home1/irteamsu/users/rklee/TheIllusionsLibraries/GANs/vanilla_gan_v1/generated_face_imgs/"
# tbt003
OUTPUT_IMAGE_SAVE_DIRECTORY = '/home1/irteamsu/rklee/TheIllusionsLibraries/GANs/vanilla_gan_v1/gen_images/'
# i7-2600k (Ubuntu)
#OUTPUT_IMAGE_SAVE_DIRECTORY = "/media/illusion/ML_Linux/temp/vanilla_gan_v1_gen_images/"
##############################################################################################
# Image Buffer Management

# image buffers
image_buffer_size = 300 
input_buff = np.empty(shape=(image_buffer_size, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3))

buff_status = []
for i in range(image_buffer_size):
    buff_status.append('empty')

current_buff_index = 0
lineIdx = 0

# load the filelist
os.chdir(INPUT_IMAGE_DIRECTORY_PATH)
jpg_files = glob.glob('*.jpg')
random.shuffle(jpg_files)

max_training_index = len(jpg_files)

exit_notification = False

def image_buffer_loader():
    global current_buff_index
    global lineIdx

    print 'image_buffer_loader'

    epoch = 0
    
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

        if (type(input_img) is not np.ndarray):
            lineIdx = lineIdx + 1
            if lineIdx >= max_training_index:
                lineIdx = 0

            print 'skip this jpg file. continue.'
            continue

        input_buff[current_buff_index] = cv2.resize(input_img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # make zero-centered
        input_buff[current_buff_index] = input_buff[current_buff_index] - PIXEL_VALUE_FOR_ZERO_CENTERED

        buff_status[current_buff_index] = 'filled'

        if lineIdx % 1000 == 0:
            print 'training_jpg_line_idx=', str(lineIdx)

        lineIdx = lineIdx + 1
        if lineIdx >= max_training_index:
            lineIdx = 0
            epoch = epoch + 1
            print 'epoch = ', str(epoch)

        current_buff_index = current_buff_index + 1
        if current_buff_index >= image_buffer_size:
            current_buff_index = 0

##############################################################################################
def main_alive_checker():
    global is_main_alive
    global exit_notification

    is_main_alive = False
    
    while True:
        if is_main_alive == False:
            # wait for the 5 secs for last chance
            time.sleep(5)
            if is_main_alive == False:
                exit_notification = True
                break;
            else:
                is_main_alive = False
        
##############################################################################################

# Launch image buffer loader
if IS_TRAINING:
    timer = threading.Timer(1, image_buffer_loader)
    timer.start()
    
    timer2 = threading.Timer(1, main_alive_checker)
    timer2.start()
##############################################################################################

class SimpleGenerator:
    def __init__(self, sess, discriminator):
        with tf.variable_scope("generator") as scope:

            self.sess = sess

            #self.z = np.random.normal(size=1000)

            # placeholder
            self.gen_X = tf.placeholder(tf.float32, [BATCH_SIZE, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT, NOISE_VECTOR_DEPTH])

            # weights
            self.gen_TRANSPOSED_CONV_W1 = tf.get_variable("GEN_TRANSPOSED_CONV_W1", shape=[4, 4, 1024, NOISE_VECTOR_DEPTH],
                                                          initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W2 = tf.get_variable("GEN_TRANSPOSED_CONV_W2", shape=[4, 4, 512, 1024],
                                                          initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W3 = tf.get_variable("GEN_TRANSPOSED_CONV_W3", shape=[4, 4, 256, 512],
                                                          initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W4 = tf.get_variable("GEN_TRANSPOSED_CONV_W4", shape=[4, 4, 128, 256],
                                                          initializer=tf.contrib.layers.xavier_initializer())
            self.gen_TRANSPOSED_CONV_W5 = tf.get_variable("GEN_TRANSPOSED_CONV_W5", shape=[4, 4, 3, 128],
                                                          initializer=tf.contrib.layers.xavier_initializer())

            # biases
            self.gen_BIAS_1 = tf.Variable(tf.zeros([1, 1024]), name="GEN_BIAS_1")
            self.gen_BIAS_2 = tf.Variable(tf.zeros([1, 512]), name="GEN_BIAS_2")
            self.gen_BIAS_3 = tf.Variable(tf.zeros([1, 256]), name="GEN_BIAS_3")
            self.gen_BIAS_4 = tf.Variable(tf.zeros([1, 128]), name="GEN_BIAS_4")
            self.gen_BIAS_5 = tf.Variable(tf.zeros([1, 3]), name="GEN_BIAS_5")

            #tf.global_variables_initializer().run()

            self.var_list_gen = [self.gen_TRANSPOSED_CONV_W1, self.gen_TRANSPOSED_CONV_W2, self.gen_TRANSPOSED_CONV_W3,
                                 self.gen_TRANSPOSED_CONV_W4, self.gen_TRANSPOSED_CONV_W5,
                                 self.gen_BIAS_1, self.gen_BIAS_2, self.gen_BIAS_3,
                                 self.gen_BIAS_4, self.gen_BIAS_5]

    def forward(self, x):
        # graph
        gen_TRANS_CONV_1 = tf.nn.relu(tf.nn.conv2d_transpose(x,
                                                              self.gen_TRANSPOSED_CONV_W1,
                                                              output_shape=[BATCH_SIZE, 8, 8, 1024],
                                                              strides=[1, 2, 2, 1],
                                                              padding="SAME") + self.gen_BIAS_1)

        gen_TRANS_CONV_2 = tf.nn.relu(tf.nn.conv2d_transpose(gen_TRANS_CONV_1,
                                                             self.gen_TRANSPOSED_CONV_W2,
                                                             output_shape=[BATCH_SIZE, 16, 16, 512],
                                                             strides=[1, 2, 2, 1],
                                                             padding="SAME") + self.gen_BIAS_2)

        gen_TRANS_CONV_3 = tf.nn.relu(tf.nn.conv2d_transpose(gen_TRANS_CONV_2,
                                                             self.gen_TRANSPOSED_CONV_W3,
                                                             output_shape=[BATCH_SIZE, 32, 32, 256],
                                                             strides=[1, 2, 2, 1],
                                                             padding="SAME") + self.gen_BIAS_3)

        gen_TRANS_CONV_4 = tf.nn.relu(tf.nn.conv2d_transpose(gen_TRANS_CONV_3,
                                                             self.gen_TRANSPOSED_CONV_W4,
                                                             output_shape=[BATCH_SIZE, 32, 32, 128],
                                                             strides=[1, 1, 1, 1],
                                                             padding="SAME") + self.gen_BIAS_4)

        gen_TRANS_CONV_5 = tf.nn.relu(tf.nn.conv2d_transpose(gen_TRANS_CONV_4,
                                                              self.gen_TRANSPOSED_CONV_W5,
                                                              output_shape=[BATCH_SIZE, 32, 32, 3],
                                                              strides=[1, 1, 1, 1],
                                                              padding="SAME") + self.gen_BIAS_5)
        
        gen_hypothesis = tf.sigmoid(gen_TRANS_CONV_5)

        # scale to 0~255
        gen_hypothesis = gen_hypothesis * 255

        # make zero centered
        gen_hypothesis = gen_hypothesis - PIXEL_VALUE_FOR_ZERO_CENTERED

        return gen_hypothesis

    def generate_fake_imgs(self):
        # Random noise vector

        noise_z = np.random.uniform(-1, 1, [BATCH_SIZE, NOISE_VECTOR_WIDTH, NOISE_VECTOR_HEIGHT, NOISE_VECTOR_DEPTH]).astype(np.float32)

        gen_hypothesis = self.forward(self.gen_X)
        network_output = self.sess.run(gen_hypothesis, feed_dict={self.gen_X: noise_z})

        fake_imgs = network_output
        self.fake_imgs = fake_imgs

        return fake_imgs

class SimpleDiscriminator:
    def __init__(self, sess):
        with tf.variable_scope("discriminator") as scope:

            self.real_img_buff = np.empty(shape=(BATCH_SIZE, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3))

            self.sess = sess

            # placeholder
            self.disc_X = tf.placeholder(tf.float32, [BATCH_SIZE, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_DEPTH])

            # weights for convolutional layers
            self.disc_CNN_W1 = tf.get_variable("DISC_CNN_W1", shape=[5, 5, 3, 16], initializer=tf.contrib.layers.xavier_initializer())
            self.disc_CNN_W2 = tf.get_variable("DISC_CNN_W2", shape=[3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
            self.disc_CNN_W3 = tf.get_variable("DISC_CNN_W3", shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())

            self.disc_FC_W1 = tf.get_variable("DISC_FC_W1", shape=[8192, 30],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.disc_FC_W2 = tf.get_variable("DISC_FC_W2", shape=[30, 30],
                                       initializer=tf.contrib.layers.xavier_initializer())
            self.disc_FC_W3 = tf.get_variable("DISC_FC_W3", shape=[30, 1],
                                       initializer=tf.contrib.layers.xavier_initializer())

            # biases
            self.disc_BIAS_CONV_1 = tf.Variable(tf.zeros([1, 16]), name="DISC_ConvBias1")
            self.disc_BIAS_CONV_2 = tf.Variable(tf.zeros([1, 32]), name="DISC_ConvBias2")
            self.disc_BIAS_CONV_3 = tf.Variable(tf.zeros([1, 32]), name="DISC_ConvBias3")
            self.disc_BIAS_FC_W1 = tf.Variable(tf.zeros([1, 30]), name="DISC_FCBias1")
            self.disc_BIAS_FC_W2 = tf.Variable(tf.zeros([1, 30]), name="DISC_FCBias2")
            self.disc_BIAS_FC_W3 = tf.Variable(tf.zeros([1, 1]), name="DISC_FCBias3")

            self.var_list_disc = [self.disc_CNN_W1, self.disc_CNN_W2, self.disc_CNN_W3,
                                  self.disc_FC_W1, self.disc_FC_W2, self.disc_FC_W3,
                                  self.disc_BIAS_CONV_1, self.disc_BIAS_CONV_2, self.disc_BIAS_CONV_3,
                                  self.disc_BIAS_FC_W1, self.disc_BIAS_FC_W2, self.disc_BIAS_FC_W3]

    def feed_images(self, fake_imgs):
        # real images are already fed
        self.fake_imgs = fake_imgs

    def forward(self, x):

        x = tf.reshape(x, [-1, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_DEPTH])

        # graph
        # 28x28
        disc_CNN1 = tf.nn.relu(tf.nn.conv2d(x, self.disc_CNN_W1, strides=[1, 1, 1, 1], padding='SAME') + self.disc_BIAS_CONV_1)
        disc_CNN2 = tf.nn.relu(tf.nn.conv2d(disc_CNN1, self.disc_CNN_W2, strides=[1, 1, 1, 1],
                                                 padding='SAME') + self.disc_BIAS_CONV_2)
        disc_CNN3 = tf.nn.relu(tf.nn.conv2d(disc_CNN2, self.disc_CNN_W3, strides=[1, 2, 2, 1],
                                                 padding='SAME') + self.disc_BIAS_CONV_3)

        # 14x14
        disc_FC_IN = tf.reshape(disc_CNN3, [-1, 16*16*32])
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
            is_main_alive = True
            
            for i in range(BATCH_SIZE):
                while buff_status[image_buff_read_index] == 'empty':
                    if exit_notification == True:
                        break

                    # print 'sleep start'
                    time.sleep(1)
                    # print 'sleep end'
                    if buff_status[image_buff_read_index] == 'filled':
                        break

                if exit_notification == True:
                    break

                np.copyto(discriminator.real_img_buff[i], input_buff[image_buff_read_index])
                # discriminator.real_img_buff[i] = input_buff[image_buff_read_index]
                buff_status[image_buff_read_index] = 'empty'

                image_buff_read_index = image_buff_read_index + 1
                if image_buff_read_index >= image_buffer_size:
                    image_buff_read_index = 0

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
            if iter % 200 == 0:
                fake_imgs = generator.generate_fake_imgs()

                # revert zero-centered
                fake_imgs = fake_imgs + PIXEL_VALUE_FOR_ZERO_CENTERED

                for j in range(len(fake_imgs)):
                    # save images to jpg files
                    filename = "iter_" + str(iter) + "_generated_" + str(j) + ".jpg"
                    cv2.imwrite(OUTPUT_IMAGE_SAVE_DIRECTORY + filename, fake_imgs[j])

    exit_notification = True

    print 'ok'
