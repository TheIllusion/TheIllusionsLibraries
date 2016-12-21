import os
import glob

f = open('training_list.txt', 'w')

# get list of jpg files
jpg_files = glob.glob( '*.jpg' )
JPG_files = glob.glob( '*.JPG' )
jpeg_files = glob.glob( '*.jpeg' )
JPEG_files = glob.glob( '*.JPEG' )
png_files = glob.glob( '*.png' )
PNG_files = glob.glob( '*.PNG' )

label = 1

for jpg_file in jpg_files:
    file_name = jpg_file[0:]
    line_string = file_name + ' ' + str(label) + '\n'
    f.write(line_string)

for jpg_file in JPG_files:
    file_name = jpg_file[0:]
    line_string = file_name + ' ' + str(label) + '\n'
    f.write(line_string)

for jpg_file in jpeg_files:
    file_name = jpg_file[0:]
    line_string = file_name + ' ' + str(label) + '\n'
    f.write(line_string)

for jpg_file in JPEG_files:
    file_name = jpg_file[0:]
    line_string = file_name + ' ' + str(label) + '\n'
    f.write(line_string)

for jpg_file in png_files:
    file_name = jpg_file[0:]
    line_string = file_name + ' ' + str(label) + '\n'
    f.write(line_string)

for jpg_file in PNG_files:
    file_name = jpg_file[0:]
    line_string = file_name + ' ' + str(label) + '\n'
    f.write(line_string)

f.close()
