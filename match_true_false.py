
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
    if float(temp[2]) > 5:
        array1[lineIdx] = True
    else:
        array1[lineIdx] = False

    temp = textData2[lineIdx].split()
    if float(temp[2]) > 0.5:
        array2[lineIdx] = True
    else:
        array2[lineIdx] = False

    #print eval_matrix[lineIdx]

count_true = 0
for lineIdx in xrange(len(textData2)):
    if array1[lineIdx] == array2[lineIdx]:
        print True
        count_true = count_true + 1
    else:
        print False

# accuracy
accuracy = float(count_true) / len(textData2)
print 'Accurary', accuracy