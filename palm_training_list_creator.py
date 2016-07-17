import os
import glob
import re

f_train = open('training_list.txt', 'w')
f_test = open('test_list.txt', 'w')

# get list of jpg files
png_files = glob.glob( '*.png' )

#label = 1
loop_idx = 0
for png_file in png_files:
    file_name = png_file[0:]
    match = re.search("_", file_name)
    if match:
        end_index = match.end()
        label = file_name[4:end_index-1]
        line_string = file_name + ' ' + str(label) + '\n'
        if (loop_idx % 10 ==  0) or (int(label) % 100 == 0):
            f_test.write(line_string)
        else:
            f_train.write(line_string)
    loop_idx = loop_idx + 1

f_train.close()
f_test.close()
