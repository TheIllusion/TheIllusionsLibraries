#
# Download image files from list files
#
# List files are assumed to be stored in _list directory
#

import os
import glob

# get list of list files
path = glob.glob( '*.txt' )

# for each item
i = 1
for list_file in path:
    name = list_file[0:-4]
    if not os.path.exists( name ):
        os.mkdir( name )
    command = 'wget -N -i %s -P %s -T 10 --tries=1'%(list_file, name)
    print "%5d"%i, name, ": ", command
    os.system( command )
    #else:
    #    print "%5d"%i, name, ": already exists"
    i += 1
