import multiprocessing
from multiprocessing.pool import Pool
#from multiprocessing.pool import ThreadPool as Pool  # to use threads

def func_1():
    return 5, 3

def func_2():
    return 4, 7

pool = Pool(processes = 2)

result1 = pool.apply_async(func_1)
result2 = pool.apply_async(func_2)

val1, val2 = result1.get()
val3, val4 = result2.get()

print 'hi'

print str(val1) + ' ' + str(val2) + ' ' + str(val3) + ' ' + str(val4)