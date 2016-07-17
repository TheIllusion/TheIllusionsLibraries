import os
import glob
import re

# get list of list files
path = glob.glob( '*' )
current_path = os.getcwd()
# for each item
i = 1
for list_file in path:
    name = list_file[0:]

    match1 = re.search(".jpg", name)
    match2 = re.search(".py", name)

    #ignore all .jpg files
    if (not match1) and (not match2):
        directory = current_path + '/' + name
        os.chdir(directory)

        # get list of jpg files in directory
        png_files = glob.glob('*.png')

        for png in png_files:
            new_png_file_name = name + '_' + png
            os.rename(png, new_png_file_name)
            print new_png_file_name

