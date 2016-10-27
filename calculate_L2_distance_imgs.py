import os
import glob
from scipy.spatial import distance

def calculate_l2_distance(img1, img2):
    l2_distance = distance.euclidean(img1,img2)
    print "L2 distance = ", str(l2_distance)
    return l2_distance

#f = open('training_list.txt', 'w')

# get list of jpg files
#jpg_files = glob.glob( '*.jpg' )

#label = 1

#for jpg_file in jpg_files:

    #file_name = jpg_file[0:]
    #line_string = file_name + ' ' + str(label) + '\n'
    #f.write(line_string)

#f.close()


a = (1,2,3)
b = (4,5,6)

calculate_l2_distance(a, b)