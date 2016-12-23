import glob
import os
import shutil

ORIGINAL_IMAGE_DIR = '/home/nhnent/H2/users/mskang/web_result/server_data/original_resized/'
PROCESSED_IMAGE_DIR = '/home/nhnent/H2/users/mskang/web_result/server_data/result_all/'

HAND_CANDIDATE_DIR = '/home/nhnent/H1/users/rklee/Data/server_data/hand_candidates/'
NONHAND_CANDIDATE_DIR = '/home/nhnent/H1/users/rklee/Data/server_data/nonhand_candidates/'

os.chdir(ORIGINAL_IMAGE_DIR)
#get oroginal jpg file lists
original_jpg_files = glob.glob( '*.jpg' )

os.chdir(PROCESSED_IMAGE_DIR)
#get processed jpg file lists
processed_jpg_files = glob.glob( '*.jpg' )

if not os.path.exists(HAND_CANDIDATE_DIR):
    os.mkdir(HAND_CANDIDATE_DIR)

if not os.path.exists(NONHAND_CANDIDATE_DIR):
    os.mkdir(NONHAND_CANDIDATE_DIR)

MAX_INDEX = 50000

index = 0

for jpg_file in original_jpg_files:
    if any(jpg_file in s for s in processed_jpg_files):
        shutil.copy2('ORIGINAL_IMAGE_DIR' + jpg_file, 'HAND_CANDIDATE_DIR' + jpg_file)
    else:
        print 'non-hand: ', jpg_file
        shutil.copy2('ORIGINAL_IMAGE_DIR' + jpg_file, 'NONHAND_CANDIDATE_DIR' + jpg_file)

    index = index + 1
    if index == MAX_INDEX:
        break