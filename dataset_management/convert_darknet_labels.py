# convert coco, bdd labels to make a unified custom dataset

import os, glob

INPUT_TXT_FILE_PATH = '/Users/Illusion/Downloads/data'
OUTPUT_PATH = '/Users/Illusion/Downloads/data_custom'

def convert_bdd_to_custom(line):
    words = line.split()

    if words[0] == '0' or \
       words[0] == '1' or \
       words[0] == '2' or \
       words[0] == '3' or \
       words[0] == '4':
        return line
    elif words[0] == '6':
        words[0] = '5'
    elif words[0] == '8':
        words[0] = '7'
    elif words[0] == '9':
        words[0] = '8'
    else:
        return None

    modified_line = ' '.join(words)
    modified_line += '\n'
    return modified_line


def convert_coco_to_custom(line):
    words = line.split()

    if words[0] == '0':
        words[0] = '4'
    elif words[0] == '1':
        words[0] = '0'
    elif words[0] == '2':
        words[0] = '2'
    elif words[0] == '3':
        words[0] = '3'
    elif words[0] == '5':
        words[0] = '1'
    elif words[0] == '6':
        words[0] = '7'
    elif words[0] == '7':
        words[0] = '8'
    elif words[0] == '9':
        words[0] = '5'
    else:
        return None

    modified_line = ' '.join(words)
    modified_line += '\n'
    return modified_line


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    txt_filelist = glob.glob(os.path.join(INPUT_TXT_FILE_PATH, '*.txt'))

    idx = 0
    for txt_file in txt_filelist:

        # open the input txt file
        f_original = open(txt_file, 'r')
        lines = f_original.readlines()

        # create an output txt file
        f = open(os.path.join(OUTPUT_PATH, os.path.basename(txt_file)), 'w+')

        for line in lines:
            modified_line = convert_bdd_to_custom(line)
            if modified_line is not None:
                f.write(modified_line)

        f_original.close()
        f.close()

        idx += 1
        if idx % 100 == 0:
            print 'idx =', idx
