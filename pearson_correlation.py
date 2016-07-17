import scipy
import math
from scipy.stats import pearsonr

#x = scipy.array([-0.65499887,  2.34644428, 2.0])
#y = scipy.array([-1.46049758,  3.86537321, 21.0])

textFile1 = open('/Users/Illusion/Temp/ava_score/original_scale/allscore.txt')
textFile2 = open('/Users/Illusion/Temp/ava_score/original_scale/sorted_evaluation_list_by_name_orig.txt')

#textFile1 = open('/Users/Illusion/Temp/ava_score/allscore_sorted_evaluation_list.txt')
#textFile2 = open('/Users/Illusion/Temp/ava_score/sorted_evaluation_list.txt')

#textFile1 = open('/Users/Illusion/Temp/ava_score/allscore_sorted_evaluation_list_by_name.txt')
#textFile2 = open('/Users/Illusion/Temp/ava_score/sorted_evaluation_list_by_name.txt')

textData1 = textFile1.readlines()
textData2 = textFile2.readlines()

array1 = [0 for x in range(len(textData2))]
array2 = [0 for x in range(len(textData2))]
temp = []
# read tokens from file
for lineIdx in xrange(len(textData2)):
    #for idx in xrange(len(textData[lineIdx])):
    temp = textData1[lineIdx].split()
    array1[lineIdx] = float(temp[2])
    temp = textData2[lineIdx].split()
    array2[lineIdx] = float(temp[2])

    #print eval_matrix[lineIdx]

r_row, p_value = pearsonr(array1, array2)

print r_row
print p_value