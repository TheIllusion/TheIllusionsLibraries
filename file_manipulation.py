#
# find '.jpg' string among each file name and convert the filename according to it
#
# List files are assumed to be stored in _list directory
#

import os
import glob
import re

# get list of list files
path = glob.glob( '*.*' )

# for each item
i = 1
for list_file in path:
    name = list_file[0:]

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
