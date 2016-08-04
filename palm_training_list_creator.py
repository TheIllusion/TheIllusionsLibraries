import os
import glob
import re

f_train = open('classification_training_list.txt', 'w')
f_test = open('classification_test_list.txt', 'w')

# get list of jpg files
png_files = glob.glob( '*.jpg' )

loopIdx = 0
for png_file in png_files:
    filename = png_file[0:]

    match = re.search("palm", filename)
    if match:
        start_index = match.end()
        match2 = re.search(".jpg", filename)
        if match2:
            end_index = match2.start()
            palm_number = filename[start_index:end_index]
            line_string = filename + '\n'
            if int(palm_number) % 20 == 0:
                f_test.write(line_string)
            else:
                f_train.write(line_string)
            if loopIdx % 1000 == 0:
                print filename
    loopIdx = loopIdx + 1

f_train.close()
f_test.close()
