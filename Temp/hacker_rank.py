# Enter your code here. Read input from STDIN. Print output to STDOUT

input_file = 'C:/Users/rokky/Desktop/Temp/input07.txt'

file = open(input_file, 'r')

# Complete the arrayManipulation function below.
def arrayManipulation(n, queries):
    lst = [0] * n
    # print lst

    print 'len(queries) =', len(queries)

    debug_idx = 0
    for q in queries:
        print 'debug_idx =', debug_idx
        debug_idx += 1
        '''
        for i in range(q[0] - 1, q[1]):
            lst[i] += q[2]
        '''

        lst[q[0]-1] += q[2]
        if q[1] < n:
            lst[q[1]] -= q[2]

    max_val = 0
    val = 0
    for i in range(n):
        val = val + lst[i]
        if val > max_val:
            max_val = val

    return max_val

if __name__ == '__main__':
    nm = file.readline().split()
    n = int(nm[0])
    m = int(nm[1])

    queries = []

    for _ in xrange(m):
        queries.append(map(int, file.readline().rstrip().split()))

    result = arrayManipulation(n, queries)

    print 'result =', result

file.close()