# rename the files that are downloaded from Getty to have appropriate .jpg names

import os
import glob
import re

# get list of list files
path = glob.glob( '*.*' )

# for each item
i = 0
for list_file in path:
    name = list_file[0:]

    # find .jpg from the file name and discard the all following characters
    '''
    match = re.search(".jpg", name)
    if match:
        end_index = match.end()
        print len(name)
        print 'end_index = ' + str(end_index)
        if len(name) > end_index:
            new_name = name[0:end_index]
            print new_name
            os.rename(name, new_name)
        else:
            print 'skipped'
    '''

    # find the specific pattern from the filename and replace it with the new pattern
    match = re.search(".txt", name)
    if match:
        new_name = name.replace('training2', '')

        if i % 100 == 0:
            print name, ' -> ', new_name

        os.rename(name, new_name)

    i = i + 1