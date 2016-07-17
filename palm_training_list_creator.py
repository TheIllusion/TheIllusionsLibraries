import os
import glob
import re

f = open('training_list.txt', 'w')

# get list of jpg files
png_files = glob.glob( '*.png' )

#label = 1

for png_file in png_files:
    file_name = png_file[0:]
    match = re.search("_", file_name)
    if match:
        end_index = match.end()
        label = file_name[4:end_index-1]
        line_string = file_name + ' ' + str(label) + '\n'
        f.write(line_string)

f.close()
