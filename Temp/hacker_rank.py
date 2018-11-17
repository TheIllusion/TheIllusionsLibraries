#!/bin/python

import math
import os
import random
import re
import sys
from datetime import timedelta, date, datetime
# Complete the time_delta function below.
def time_delta(t1, t2):

    day1 = datetime(2015, 5, 10, 13, 54, 36, 0, )
    day2 = datetime(2015, 5, 10, 13, 54, 36, 0, datetime.timezone(timedelta(hours=0)))

    if day1 > day2:
        delta = day1 - day2
    else:
        delta = day2 - day1

    print int(delta.total_seconds())

    t1_list = t1.split()
    t2_list = t2.split()

    return '100'

'''
2
Sun 10 May 2015 13:54:36 -0700
Sun 10 May 2015 13:54:36 -0000
Sat 02 May 2015 19:54:36 +0530
Fri 01 May 2015 13:54:36 -0000
'''

if __name__ == '__main__':
    #fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(raw_input())

    for t_itr in xrange(t):
        t1 = raw_input()

        t2 = raw_input()

        delta = time_delta(t1, t2)

        print delta + '\n'
