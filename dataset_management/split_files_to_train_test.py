import os
import glob
import re
import random
import shutil

# get list of list files
source_dir = '/Users/Illusion/Temp/20150809'
target_train_dir = '/Users/Illusion/Temp/train'
target_test_dir = '/Users/Illusion/Temp/test'

if not os.path.exists(source_dir):
    exit(-1)

if not os.path.exists(target_train_dir):
    exit(-1)

if not os.path.exists(target_test_dir):
    exit(-1)

file_list = glob.glob(os.path.join(source_dir, '*.*'))

random.shuffle(file_list)

# for each item
for idx, filename in enumerate(file_list):

    if idx % 10 == 0:
        shutil.move(filename, os.path.join(target_test_dir, os.path.basename(filename)))
        print 'idx =', idx
    else:
        shutil.move(filename, os.path.join(target_train_dir, os.path.basename(filename)))