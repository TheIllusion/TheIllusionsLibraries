import numpy as np
import os, glob, random, re, time, threading
import cv2

# t005 
# tiny dataset for SR (div2k,4X.)
ANSWER_IMAGE_DIRECTORY_PATH = "/home1/irteamsu/rklee/tiny_dataset/sr/div2k_custom_tr_set_for_x4/DIV2K_train_HR_modified/"

INPUT_IMAGE_DIRECTORY_PATH = "/home1/irteamsu/rklee/tiny_dataset/sr/div2k_custom_tr_set_for_x4/X4_modified/"

IS_TRAINING = True

##############################################################################################
# Image Buffer Management

INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# image buffers
image_buffer_size = 30

# OpenCV format
# input_buff = np.empty(shape=(image_buffer_size, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3))
# answer_buff = np.empty(shape=(image_buffer_size, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3))

# PyTorch format
input_buff = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

buff_status = []
for i in range(image_buffer_size):
    buff_status.append('empty')

current_buff_index = 0
lineIdx = 0

# load the filelist
# os.chdir(ANSWER_IMAGE_DIRECTORY_PATH)
os.chdir(INPUT_IMAGE_DIRECTORY_PATH)
jpg_files = glob.glob('*.png')
random.shuffle(jpg_files)

max_training_index = len(jpg_files)

exit_notification = False

def image_buffer_loader():
    global current_buff_index
    global lineIdx
    global exit_notification

    print 'image_buffer_loader'

    epoch = 0

    while True:
        filename_ = jpg_files[lineIdx]

        end_index = 0

        match = re.search(".png", filename_)
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

        while_start_time = time.time()
        while buff_status[current_buff_index] == 'filled':
            if exit_notification == True:
                break

            # print 'sleep start'
            time.sleep(1)
            # print 'sleep end'
            if buff_status[current_buff_index] == 'empty':
                break

        if exit_notification == True:
            print 'Exit(1)'
            break

        # Input Image
        filename = os.path.join(INPUT_IMAGE_DIRECTORY_PATH, training_file_name)
        input_img = cv2.imread(filename, cv2.IMREAD_COLOR)

        if (type(input_img) is not np.ndarray):
            lineIdx = lineIdx + 1
            if lineIdx >= max_training_index:
                lineIdx = 0

            print 'skip this jpg file. continue. filename=', filename
            continue

        '''
            input_buff[current_buff_index] = cv2.resize(input_img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT),
                                                        interpolation=cv2.INTER_LINEAR)
        '''
        input_img_tmp = cv2.resize(input_img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)

        input_img_tmp = input_img_tmp[..., [2, 1, 0]]
        input_buff[current_buff_index][0, :, :] = input_img_tmp[:, :, 0]
        input_buff[current_buff_index][1, :, :] = input_img_tmp[:, :, 1]
        input_buff[current_buff_index][2, :, :] = input_img_tmp[:, :, 2]

        # Answer Image
        filename = os.path.join(ANSWER_IMAGE_DIRECTORY_PATH, training_file_name)
        answer_img = cv2.imread(filename, cv2.IMREAD_COLOR)

        if (type(answer_img) is not np.ndarray):
            lineIdx = lineIdx + 1
            if lineIdx >= max_training_index:
                lineIdx = 0

            print 'skip this answer jpg file. continue. filename=', filename
            continue

        answer_img_tmp = cv2.resize(answer_img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)

        answer_img_tmp = answer_img_tmp[..., [2, 1, 0]]
        answer_buff[current_buff_index][0, :, :] = answer_img_tmp[:, :, 0]
        answer_buff[current_buff_index][1, :, :] = answer_img_tmp[:, :, 1]
        answer_buff[current_buff_index][2, :, :] = answer_img_tmp[:, :, 2]

        buff_status[current_buff_index] = 'filled'

        if lineIdx % 1000 == 0:
            print 'training_jpg_line_idx =', str(lineIdx), 'epoch =', str(epoch)

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
            # wait for the 7 secs for last chance
            time.sleep(7)
            if is_main_alive == False:
                exit_notification = True
                print 'Exit(2)'
                break
            else:
                is_main_alive = False


##############################################################################################

# Launch image buffer loader
if IS_TRAINING:
    timer = threading.Timer(1, image_buffer_loader)
    timer.start()

    timer2 = threading.Timer(1, main_alive_checker)
    timer2.start()

###############################################################################################
