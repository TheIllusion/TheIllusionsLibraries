import numpy as np
import os, glob, random, re, time, threading
import cv2

IS_TRAINING = False

##############################################################################################
# directory settings 

# tbt005 (10.161.31.83)
INPUT_IMAGE_DIRECTORY_PATH = "/data/rklee/hair_dyeing/black_to_blonde/trainA/"
ANSWER_IMAGE_DIRECTORY_PATH_BLONDE = "/data/rklee/hair_dyeing/black_to_blonde/trainB/"
ANSWER_IMAGE_DIRECTORY_PATH_BROWN = '/home1/irteamsu/data/rklee/hair_dyeing/black2brown/trainB/'
ANSWER_IMAGE_DIRECTORY_PATH_WINE = '/home1/irteamsu/data/rklee/hair_dyeing/black_to_wine_female/trainB/'
ANSWER_IMAGE_DIRECTORY_PATH_CYAN = '/home1/irteamsu/data/rklee/hair_dyeing/black_to_cyan_female/trainB/'
ANSWER_IMAGE_DIRECTORY_PATH_BLUE = '/home1/irteamsu/data/rklee/hair_dyeing/black_to_blue_female/trainB/'
ANSWER_IMAGE_DIRECTORY_PATH_SILVER = '/home1/irteamsu/data/rklee/hair_dyeing/black_to_silver_female/trainB/'
ANSWER_IMAGE_DIRECTORY_PATH_RED = '/home1/irteamsu/data/rklee/hair_dyeing/black_to_pure_red_female/trainB/'
ANSWER_IMAGE_DIRECTORY_PATH_ORANGE = '/home1/irteamsu/data/rklee/hair_dyeing/black_to_red_orange_female/trainB/'
ANSWER_IMAGE_DIRECTORY_PATH_GRAY = '/home1/irteamsu/data/rklee/hair_dyeing/black2gray/trainB/'

##############################################################################################
# Image Buffer Management

INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

# image buffers
image_buffer_size = 100

# OpenCV format
# input_buff = np.empty(shape=(image_buffer_size, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3))
# answer_buff = np.empty(shape=(image_buffer_size, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3))

# Hair Color List
hair_color_list = ['BLONDE', 'WINE', 'BROWN', 'CYAN', 'BLUE', 'SILVER', 'RED', 'ORANGE', 'GRAY']
#hair_color_list = ['BLONDE']

##############################################################################################
# input buffers
# PyTorch format
input_buff = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

# answer buffers
answer_buff_blonde = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff_wine = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff_brown = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff_cyan = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff_blue = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff_silver = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff_red = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff_orange = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
answer_buff_gray = np.empty(shape=(image_buffer_size, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

##############################################################################################
buff_status = []

for i in range(image_buffer_size):
    buff_status.append('empty')

current_buff_index = 0

lineIdxInput = 0

lineIdxAnswer_blonde = 0
lineIdxAnswer_wine = 0
lineIdxAnswer_brown = 0
lineIdxAnswer_cyan = 0
lineIdxAnswer_blue = 0
lineIdxAnswer_silver = 0
lineIdxAnswer_red = 0
lineIdxAnswer_orange = 0
lineIdxAnswer_gray = 0
##############################################################################################
# load the filelist

# input (probably black)
os.chdir(INPUT_IMAGE_DIRECTORY_PATH)
jpg_files_input = glob.glob('*.jpg')
random.shuffle(jpg_files_input)

# blonde
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_BLONDE)
jpg_files_answer_blonde = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_blonde)

# wine
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_WINE)
jpg_files_answer_wine = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_wine)

# brown
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_BROWN)
jpg_files_answer_brown = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_brown)

# cyan
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_CYAN)
jpg_files_answer_cyan = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_cyan)

# blue
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_BLUE)
jpg_files_answer_blue = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_blue)

# silver
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_SILVER)
jpg_files_answer_silver = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_silver)

# red
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_RED)
jpg_files_answer_red = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_red)

# orange
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_ORANGE)
jpg_files_answer_orange = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_orange)

# gray
os.chdir(ANSWER_IMAGE_DIRECTORY_PATH_GRAY)
jpg_files_answer_gray = glob.glob('*.jpg')
random.shuffle(jpg_files_answer_gray)

# estimate the length of the file lists
max_training_index_input = len(jpg_files_input)
max_training_index_answer_blonde = len(jpg_files_answer_blonde)
max_training_index_answer_wine = len(jpg_files_answer_wine)
max_training_index_answer_brown = len(jpg_files_answer_brown)
max_training_index_answer_cyan = len(jpg_files_answer_cyan)
max_training_index_answer_blue = len(jpg_files_answer_blue)
max_training_index_answer_silver = len(jpg_files_answer_silver)
max_training_index_answer_red = len(jpg_files_answer_red)
max_training_index_answer_orange = len(jpg_files_answer_orange)
max_training_index_answer_gray = len(jpg_files_answer_gray)

##############################################################################################
# Dictionary for answer buffers 
answer_buff_dict = {}
answer_buff_dict['BLONDE'] = answer_buff_blonde
answer_buff_dict['WINE'] = answer_buff_wine
answer_buff_dict['BROWN'] = answer_buff_brown
answer_buff_dict['CYAN'] = answer_buff_cyan
answer_buff_dict['BLUE'] = answer_buff_blue
answer_buff_dict['SILVER'] = answer_buff_silver
answer_buff_dict['RED'] = answer_buff_red
answer_buff_dict['ORANGE'] = answer_buff_orange
answer_buff_dict['GRAY'] = answer_buff_gray

jpg_files_answer_dict = {}
jpg_files_answer_dict['BLONDE'] = jpg_files_answer_blonde
jpg_files_answer_dict['WINE'] = jpg_files_answer_wine
jpg_files_answer_dict['BROWN'] = jpg_files_answer_brown
jpg_files_answer_dict['CYAN'] = jpg_files_answer_cyan
jpg_files_answer_dict['BLUE'] = jpg_files_answer_blue
jpg_files_answer_dict['SILVER'] = jpg_files_answer_silver
jpg_files_answer_dict['RED'] = jpg_files_answer_red
jpg_files_answer_dict['ORANGE'] = jpg_files_answer_orange
jpg_files_answer_dict['GRAY'] = jpg_files_answer_gray

lineIdxAnswer_dict = {}
lineIdxAnswer_dict['BLONDE'] = lineIdxAnswer_blonde
lineIdxAnswer_dict['WINE'] = lineIdxAnswer_wine
lineIdxAnswer_dict['BROWN'] = lineIdxAnswer_brown
lineIdxAnswer_dict['CYAN'] = lineIdxAnswer_cyan
lineIdxAnswer_dict['BLUE'] = lineIdxAnswer_blue
lineIdxAnswer_dict['SILVER'] = lineIdxAnswer_silver
lineIdxAnswer_dict['RED'] = lineIdxAnswer_red
lineIdxAnswer_dict['ORANGE'] = lineIdxAnswer_orange
lineIdxAnswer_dict['GRAY'] = lineIdxAnswer_gray

max_training_index_answer_dict = {}
max_training_index_answer_dict['BLONDE'] = max_training_index_answer_blonde
max_training_index_answer_dict['WINE'] = max_training_index_answer_wine
max_training_index_answer_dict['BROWN'] = max_training_index_answer_brown
max_training_index_answer_dict['CYAN'] = max_training_index_answer_cyan
max_training_index_answer_dict['BLUE'] = max_training_index_answer_blue
max_training_index_answer_dict['SILVER'] = max_training_index_answer_silver
max_training_index_answer_dict['RED'] = max_training_index_answer_red
max_training_index_answer_dict['ORANGE'] = max_training_index_answer_orange
max_training_index_answer_dict['GRAY'] = max_training_index_answer_gray

answer_directory_dict = {}
answer_directory_dict['BLONDE'] = ANSWER_IMAGE_DIRECTORY_PATH_BLONDE
answer_directory_dict['WINE'] = ANSWER_IMAGE_DIRECTORY_PATH_WINE
answer_directory_dict['BROWN'] = ANSWER_IMAGE_DIRECTORY_PATH_BROWN
answer_directory_dict['CYAN'] = ANSWER_IMAGE_DIRECTORY_PATH_CYAN
answer_directory_dict['BLUE'] = ANSWER_IMAGE_DIRECTORY_PATH_BLUE
answer_directory_dict['SILVER'] = ANSWER_IMAGE_DIRECTORY_PATH_SILVER
answer_directory_dict['RED'] = ANSWER_IMAGE_DIRECTORY_PATH_RED
answer_directory_dict['ORANGE'] = ANSWER_IMAGE_DIRECTORY_PATH_ORANGE
answer_directory_dict['GRAY'] = ANSWER_IMAGE_DIRECTORY_PATH_GRAY

# temporary filenames for each iteration
training_file_name_answer_dict = {}

##############################################################################################

exit_notification = False

def image_buffer_loader():
    global current_buff_index
    global lineIdxInput
    global exit_notification

    print 'image_buffer_loader'

    epoch = 0

    while True:
        
        ################################################################
        # read an input image filename
        filename_input_ = jpg_files_input[lineIdxInput]

        end_index = 0

        match = re.search(".jpg", filename_input_)
        if match:
            end_index = match.end()
            filename_input = filename_input_[0:end_index]

        if end_index == 0:
            lineIdxInput = lineIdxInput + 1
            if lineIdxInput >= max_training_index_input:
                lineIdxInput = 0

            print 'skip this input jpg file. continue.'
            continue

        training_file_name_input = filename_input

        ################################################################
        # read an answer image filename (iterate through the colors)
        
        for color in hair_color_list:

            #filename_answer_ = jpg_files_answer_blonde[lineIdxAnswer_BLONDE]
            file_list = jpg_files_answer_dict[color]
            filename_answer_ = file_list[lineIdxAnswer_dict[color]]

            end_index = 0

            match = re.search(".jpg", filename_answer_)
            if match:
                end_index = match.end()
                filename_answer = filename_answer_[0:end_index]

            if end_index == 0:
                lineIdxAnswer_dict[color] = lineIdxAnswer_dict[color] + 1
                
                if lineIdxAnswer_dict[color] >= max_training_index_answer_dict[color]:
                    lineIdxAnswer_dict[color] = 0

                print 'skip this answer jpg file. continue. color =', color
                continue

            training_file_name_answer_dict[color] = filename_answer
        
        ################################################################

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

        #############################################################################
        # Input Image
        filename = os.path.join(INPUT_IMAGE_DIRECTORY_PATH, training_file_name_input)
        input_img = cv2.imread(filename, cv2.IMREAD_COLOR)

        if (type(input_img) is not np.ndarray):
            lineIdxInput = lineIdxInput + 1
            if lineIdxInput >= max_training_index_input:
                lineIdxInput = 0

            print '(cannot read) skip this input jpg file. continue.'
            continue

        input_img_tmp = cv2.resize(input_img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)

        input_img_tmp = input_img_tmp[..., [2,1,0]]
        input_buff[current_buff_index][0, :, :] = input_img_tmp[:, :, 0]
        input_buff[current_buff_index][1, :, :] = input_img_tmp[:, :, 1]
        input_buff[current_buff_index][2, :, :] = input_img_tmp[:, :, 2]

        #############################################################################
        # Answer Image (iterate through the colors)
        
        for color in hair_color_list:
            filename = os.path.join(answer_directory_dict[color], training_file_name_answer_dict[color])
            answer_img = cv2.imread(filename, cv2.IMREAD_COLOR)

            if (type(answer_img) is not np.ndarray):
                lineIdxAnswer_dict[color] = lineIdxAnswer_dict[color] + 1
                if lineIdxAnswer_dict[color] >= max_training_index_answer_dict[color]:
                    lineIdxAnswer_dict[color] = 0

                print '(cannot read) skip this answer jpg file. continue. color =', color
                continue

            answer_img_tmp = cv2.resize(answer_img, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT),
                                       interpolation=cv2.INTER_LINEAR)

            answer_img_tmp = answer_img_tmp[..., [2, 1, 0]]
            
            answer_buff = answer_buff_dict[color]
            answer_buff[current_buff_index][0, :, :] = answer_img_tmp[:, :, 0]
            answer_buff[current_buff_index][1, :, :] = answer_img_tmp[:, :, 1]
            answer_buff[current_buff_index][2, :, :] = answer_img_tmp[:, :, 2]

        buff_status[current_buff_index] = 'filled'

        #############################################################################
        if lineIdxInput % 300 == 0:
            print 'jpg_idx_input =', str(lineIdxInput), 'epoch =', str(epoch)
            
            for color in hair_color_list:
                print 'jpg_idx_answer_' + color, '= ' + str(lineIdxAnswer_dict[color]), 'epoch =', str(epoch)
                
        #############################################################################
        # increment the input index
        lineIdxInput = lineIdxInput + 1
        if lineIdxInput >= max_training_index_input:
            lineIdxInput = 0
            epoch = epoch + 1
            print 'epoch = ', str(epoch)

        #############################################################################
        # increment the answers index (iterate through the colors)
        for color in hair_color_list:
            lineIdxAnswer_dict[color] = lineIdxAnswer_dict[color] + 1
            if lineIdxAnswer_dict[color] >= max_training_index_answer_dict[color]:
                lineIdxAnswer_dict[color] = 0

        #############################################################################
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
