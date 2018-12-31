import os
import re
import shutil

PATH = '/Users/Illusion/Downloads/sample_results_3/'
#PATH = '/Users/Illusion/Temp/output/'
OUTPUT_PATH = '/Users/Illusion/Downloads/sample_results_filename_modified/'

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

# ordinary os.walk (sample code)
'''
for root, dirs, files in os.walk(PATH, topdown=False):
   for name in files:
      print(os.path.join(root, name))
   for name in dirs:
      print(os.path.join(root, name))
'''

# copy files to the OUTPUT_PATH (filename will be modified to include the parent direcoty names)
for root, dirs, files in os.walk(PATH, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        fullpath_filename = os.path.join(root, name)

        result = re.search(PATH, fullpath_filename)
        #print result.end()
        filename = fullpath_filename[result.end():]
        filename = re.sub('[/]', '_', filename)

        print filename

        shutil.copy2(fullpath_filename, os.path.join(OUTPUT_PATH, filename))

    '''
    for name in dirs:
        print(os.path.join(root, name))
    '''
