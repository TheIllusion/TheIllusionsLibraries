#!/bin/python

import math
import os
import random
import re
import sys
#from itertools import groupby

if __name__ == '__main__':
    s = raw_input()
    #s = sys.stdin.readline()

    # input = list(s)

    char_dict = {}
    value_list = set()
    '''
    for k, i in groupby(s):
        # print list(k)[0], len(list(i))
        val = len(list(i))
        char = list(k)[0]
        char_dict[char] = val
        value_list.add(val)
    '''

    for char in s:
        if not (char in char_dict.keys()):
            char_dict[char] = 1
        else:
            char_dict[char] += 1

    for key in char_dict:
        value_list.add(char_dict[key])

    value_list = list(value_list)
    value_list.sort(reverse=True)

    # starts from the maximum value
    chosen_keys = []
    idx = 0
    for current_max in value_list:
        # print 'current_max=', current_max
        for key in char_dict.keys():
            if char_dict[key] == current_max:
                chosen_keys.append(key)

        chosen_keys.sort()

        for k in chosen_keys:
            print k, char_dict[k]
            idx += 1
            # print 'idx=', idx
            if idx == 3:
                break

        chosen_keys = []

        if idx == 3:
            break

    # print chosen_keys

    #print char_dict
    #print value_list




