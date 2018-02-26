# get the filesize variations according to mozjpeg's quality factor

import glob
import os

TEST_ROOT_PATH = '/Users/Illusion/Documents/Data/shopping_mall/'

ORIGINAL_IMAGE_PATH = os.path.join(TEST_ROOT_PATH, 'mini_testset')

# mozjpeg
moz_q50_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q50')
moz_q55_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q55')
moz_q60_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q60')
moz_q65_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q65')
moz_q70_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q70')
moz_q75_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q75')
moz_q80_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q80')
moz_q85_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q85')
moz_q90_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q90')
moz_q95_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q95')
moz_q100_path = os.path.join(TEST_ROOT_PATH, 'mini_testset_mozjpeg/mini_testset_mozjpeg_q100')

# filelist
if os.path.exists(ORIGINAL_IMAGE_PATH):
    os.chdir(ORIGINAL_IMAGE_PATH)
    img_list = glob.glob('*.jpg')
    # img_list.sort()

else:
    print 'original image directory does not exist!'
    exit(0)

# directory list
dir_list = []
dir_list.append(ORIGINAL_IMAGE_PATH)
dir_list.append(moz_q100_path)
dir_list.append(moz_q95_path)
dir_list.append(moz_q90_path)
dir_list.append(moz_q85_path)
dir_list.append(moz_q80_path)
dir_list.append(moz_q75_path)
dir_list.append(moz_q70_path)
dir_list.append(moz_q65_path)
dir_list.append(moz_q60_path)
dir_list.append(moz_q55_path)
dir_list.append(moz_q50_path)

#print dir_list

# get file size
def get_file_size(full_path):
    try:
        size = os.path.getsize(full_path)
        if size != 0:
            size = size / 1000
        return size
    except:
        return 0

# dictionary for filesize (filename : filesize list)
filesize_list_dict = {}

for file in img_list:

    filesize_list = []
    filesize_percent_list = []

    # use filesize
    #filesize_list_dict[file] = filesize_list
    # use percentage
    filesize_list_dict[file] = filesize_percent_list

    for dir in dir_list:

        filesize = get_file_size(os.path.join(dir, file))

        # use file-size
        filesize_list.append(filesize)

        if len(filesize_percent_list) == 0:
            filesize_percent_list.append(100)
        else:
            percent_to_original = round((float(filesize) * 100) / filesize_list[0])
            filesize_percent_list.append(percent_to_original)

        # use percentage compared to the original image

print filesize_list_dict