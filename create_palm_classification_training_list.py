# open training_list.txt and then modify it (add the type information at the end of each line)

import re
training_list_file_name = '/Users/Illusion/Documents/Data/palm_data/NHN_palms/NHN_palm_aligned_Marking_Result/Result_Saengmyoung/resized/classification_training_list.txt'
classification_type_file_name = '/Users/Illusion/Documents/Data/palm_data/NHN_palms/NHN_palm_aligned_Marking_Result/Result_Saengmyoung/types_Saengmyoung.txt'
edited_list_file_name = '/Users/Illusion/Documents/Data/palm_data/NHN_palms/NHN_palm_aligned_Marking_Result/Result_Saengmyoung/training_list_classification_Saengmyoung.txt'

training_list_file = open(training_list_file_name)
training_list = training_list_file.readlines()
max_training_index = len(training_list)

# Assume that this file has a finite length of 780 (from 1 to 781)
classification_type_file = open(classification_type_file_name)
classification_list = classification_type_file.readlines()

edited_training_list_file = open(edited_list_file_name, 'w')

for lineIdx in xrange(len(training_list)):
    filename_ = training_list[lineIdx]
    filename = filename_.replace('\n', '')

    match = re.search("palm", filename)
    if match:
        start_index = match.end()
        match2 = re.search(".jpg", filename)
        if match2:
            end_index = match2.start()
            palm_number = filename[start_index:end_index]
            if lineIdx % 1000 == 0:
                print palm_number
            new_line_string = filename + ' ' + classification_list[int(palm_number)][-2:-1] + '\n'
            edited_training_list_file.write(new_line_string)

training_list_file.close()
edited_training_list_file.close()
classification_type_file.close()