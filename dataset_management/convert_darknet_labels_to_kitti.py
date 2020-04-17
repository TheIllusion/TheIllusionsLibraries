# convert darknet(yolov3) result labels to kitti evaluation format

import os, glob

INPUT_TXT_FILE_PATH = '/Users/Illusion/Downloads/txt_darknet_format'
OUTPUT_PATH = '/Users/Illusion/Downloads/txt_kitti_format'

def convert_dark_to_kitti(line):
    words = line.split()
    result_words = []

    if words[0] == 'car' or \
       words[0] == 'truck':

        if words[0] == 'car':
            type = 'Car'
        else:
            type = 'Truck'

        truncated = '0.0'
        occluded = '0'
        alpha = '2.00'
        left = words[2] + '.00'
        top = words[3] + '.00'
        right = words[4] + '.00'
        bottom = words[5] + '.00'
        dim_h = '1.69'
        dim_w = '1.62'
        dim_l = '3.99'
        rem_0 = '-24.05'
        rem_1 = '2.74'
        rem_2 = '47.17'
        rem_3 = '1.69'
        prob = str("{0:.2f}".format(float(words[1])))

        result_words.append(type)
        result_words.append(truncated)
        result_words.append(occluded)
        result_words.append(alpha)
        result_words.append(left)
        result_words.append(top)
        result_words.append(right)
        result_words.append(bottom)
        result_words.append(dim_h)
        result_words.append(dim_w)
        result_words.append(dim_l)
        result_words.append(rem_0)
        result_words.append(rem_1)
        result_words.append(rem_2)
        result_words.append(rem_3)
        result_words.append(prob)
    else:
        return None

    modified_line = ' '.join(result_words)
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
            modified_line = convert_dark_to_kitti(line)
            if modified_line is not None:
                f.write(modified_line)

        f_original.close()
        f.close()

        idx += 1
        if idx % 100 == 0:
            print 'idx =', idx
