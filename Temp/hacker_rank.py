#!/bin/python

import math
import os
import random
import re
import sys

# Complete the time_delta function below.
def time_delta(t1, t2):
    t1_list = t1.split()
    t2_list = t2.split()

    t1_time_zone = t1_list[5]
    t2_time_zone = t2_list[5]

    print 't1_time_zone:', t1_time_zone
    print 't2_time_zone:', t2_time_zone
    return '100'

if __name__ == '__main__':
    #fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(raw_input())

    for t_itr in xrange(t):
        t1 = raw_input()

        t2 = raw_input()

        delta = time_delta(t1, t2)

        print delta + '\n'

'''
2
Sun 10 May 2015 13:54:36 -0700
Sun 10 May 2015 13:54:36 -0000
Sat 02 May 2015 19:54:36 +0530
Fri 01 May 2015 13:54:36 -0000
'''