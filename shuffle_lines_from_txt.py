# shuffle lines randomly from the text file

import random
lines = open('training_list.txt').readlines()
random.shuffle(lines)
open('shuffle_training_list.txt', 'w').writelines(lines)
