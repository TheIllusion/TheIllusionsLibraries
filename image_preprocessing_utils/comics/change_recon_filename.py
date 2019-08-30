# -*- coding: utf-8 -*-

import os
import glob
from shutil import copyfile
import re

SOURCE_DIR = '/Users/Illusion/Temp/00007_hannar'
OUTPUT_DIR = '/Users/Illusion/Temp/00007_hannar_OUT'

def extract_core_filename(original_filename):

    numbers = re.findall(r'\d+-\d+-\d+-\d+', original_filename)
    title_name = re.findall(r'^\w+-', original_filename)

    if (len(numbers) > 0 and len(title_name) > 0):
        core_filename = title_name[0] + numbers[0] + original_filename[-4:]
    else:
        # fail case. return the original filename.
        core_filename = original_filename

    return core_filename


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    filenames = glob.glob(os.path.join(SOURCE_DIR, "*.*"))

    for file in filenames:
        core_filename = extract_core_filename(os.path.basename(file))

        print core_filename

        copyfile(file, os.path.join(OUTPUT_DIR, core_filename))

