import os
import glob

f = open('training_list.txt', 'w')

# get list of jpg files
jpg_files = glob.glob( '*.jpg' )

label = 1

for jpg_file in jpg_files:
    file_name = jpg_file[0:]
    line_string = file_name + ' ' + str(label) + '\n'
    f.write(line_string)

f.close()
