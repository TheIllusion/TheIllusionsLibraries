import os, glob
import cv2
import numpy as np
import random

RESULT_IMAGE_DIRECTORY = '/Users/Illusion/Documents/Data/gans_for_video/mug_concat_custom/happiness/'

# result_image = first_frame + 20 sampled future frames
def create_concat_imgs_in_directory(directory_path):

    if os.path.exists(directory_path):
        os.chdir(directory_path)
    else:
        return False

    jpg_filelist = glob.glob('*.jpg')

    list_length = len(jpg_filelist)
    if list_length < 60:
        print 'the length of the file list is less than 60. return False.'
        return False

    sample_interval = int(float(list_length) / 30)

    if not os.path.exists(RESULT_IMAGE_DIRECTORY):
        os.mkdir(RESULT_IMAGE_DIRECTORY)

    jpg_filelist.sort(reverse=False)

    first_jpg_file = jpg_filelist[0]

    first_jpg_img = cv2.imread(first_jpg_file, cv2.IMREAD_UNCHANGED)

    frames_list = []
    frames_list.append(first_jpg_img)

    random_num = random.randrange(0, 1000000000)

    for i in range(1, 21):
        jpg_file = jpg_filelist[i * sample_interval]

        current_jpg_img = cv2.imread(jpg_file, cv2.IMREAD_UNCHANGED)

        frames_list.append(current_jpg_img)

    concated_img = np.hstack(tuple(frames_list))

    # save the result
    cv2.imwrite(RESULT_IMAGE_DIRECTORY + 'concat_happy_' + str(random_num) + '_' + jpg_file, concated_img)

    return True

if __name__ == '__main__':

    happy_dir_list = glob.glob('/Users/Illusion/Documents/Data/gans_for_video/MUG/subjects3/*/happiness/take*')
    happy_dir_list = happy_dir_list + glob.glob('/Users/Illusion/Documents/Data/gans_for_video/MUG/subjects3/*/*/happiness/take*')

    print 'len(happy_dir_list) =', len(happy_dir_list)

    for dir in happy_dir_list:

        print 'processing. dir =', dir

        ret = create_concat_imgs_in_directory(dir)

        if ret == False:
            print 'Error occurred!! directory name =', dir
            #break

    print 'done'