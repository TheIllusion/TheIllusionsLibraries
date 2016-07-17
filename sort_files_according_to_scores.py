import math

#textFile = open('/Users/Illusion/Temp/ava_score//evaluation_list.txt')
#textFile = open('/Users/Illusion/Temp/ava_score/allscore.txt')

#outFile = open('/Users/Illusion/Temp/ava_score/sorted_evaluation_list.txt', 'w')
#outFile = open('/Users/Illusion/Temp/ava_score/allscore_sorted_evaluation_list.txt', 'w')

#textFile = open('/Users/Illusion/Temp/ava_score/allscore_sorted_evaluation_list_rescaled.txt')
#outFile = open('/Users/Illusion/Temp/ava_score/allscore_sorted_evaluation_list_by_name.txt', 'w')

textFile = open('/Users/Illusion/Temp/ava_score/allscore.txt')
outFile = open('/Users/Illusion/Temp/ava_score/allscore_sorted.txt', 'w')

textData = textFile.readlines()

# Creates a list containing 5 lists, each of 8 items, all set to 0
w, h = 3, len(textData)
eval_matrix = [[0 for x in range(w)] for y in range(h)]
sorted_eval_matrix = [[0 for x in range(w)] for y in range(h)]

# read tokens from file
for lineIdx in xrange(len(textData)):
    #for idx in xrange(len(textData[lineIdx])):
    eval_matrix[lineIdx] = textData[lineIdx].split()
    #eval_matrix[lineIdx][0] = eval_matrix[lineIdx][0].zfill(10)
    #print eval_matrix[lineIdx]

# sort the matrix
sorted_eval_matrix = sorted(eval_matrix, key=lambda x: x[2], reverse=True)
#sorted_eval_matrix = sorted(eval_matrix, key=lambda x: x[0], reverse=False)

# save the matrix to the file
for lineIdx in xrange(len(textData)):
    '''
    outFile.write('<img src = "http://10.77.29.155/H2/AVAdataset/AVA_dataset/fooddrink/')
    outFile.write(sorted_eval_matrix[lineIdx][0])
    outFile.write('"> <p> (Bad:')
    outFile.write(sorted_eval_matrix[lineIdx][1])
    outFile.write(') (Good:')
    outFile.write(sorted_eval_matrix[lineIdx][2])
    outFile.write(') </p>')
    outFile.write('\n')
    '''

    # Equalize the values
    '''
    if lineIdx < (len(textData)/10):
        sorted_eval_matrix[lineIdx][2] = str(10)
    elif lineIdx < 2*(len(textData)/10):
        sorted_eval_matrix[lineIdx][2] = str(9)
    elif lineIdx < 3 * (len(textData) / 10):
        sorted_eval_matrix[lineIdx][2] = str(8)
    elif lineIdx < 4 * (len(textData) / 10):
        sorted_eval_matrix[lineIdx][2] = str(7)
    elif lineIdx < 5 * (len(textData) / 10):
        sorted_eval_matrix[lineIdx][2] = str(6)
    elif lineIdx < 6 * (len(textData) / 10):
        sorted_eval_matrix[lineIdx][2] = str(5)
    elif lineIdx < 7 * (len(textData) / 10):
        sorted_eval_matrix[lineIdx][2] = str(4)
    elif lineIdx < 8 * (len(textData) / 10):
        sorted_eval_matrix[lineIdx][2] = str(3)
    elif lineIdx < 9 * (len(textData) / 10):
        sorted_eval_matrix[lineIdx][2] = str(2)
    else:
        sorted_eval_matrix[lineIdx][2] = str(1)
    '''

    outFile.write(sorted_eval_matrix[lineIdx][0])
    outFile.write(' ')
    outFile.write(sorted_eval_matrix[lineIdx][1])
    outFile.write(' ')
    outFile.write(sorted_eval_matrix[lineIdx][2])
    outFile.write('\n')