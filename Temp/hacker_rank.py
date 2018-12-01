# Enter your code here. Read input from STDIN. Print output to STDOUT

input_file = 'C:/Users/rokky/Desktop/Temp/inputs.txt'

file = open(input_file, 'r')

#test_cases = int(raw_input())

test_cases = int(file.readline().split()[0])

def erase_one_element(current_sidelength, inputs):
    # print 'inputs =', inputs

    if current_sidelength < max(inputs[0], inputs[len(inputs) - 1]):
        print 'current_sidelength=', current_sidelength
        print 'inputs[0], inputs[-1]', inputs[0], inputs[-1]
        return current_sidelength, inputs

    updated_sidelength = max(inputs[0], inputs[len(inputs) - 1])

    if len(inputs) <= 1:
        return updated_sidelength, []

    if inputs[0] > inputs[len(inputs) - 1]:
        #updated_inputs = inputs[1:]
        inputs.pop(0)
    else:
        #updated_inputs = inputs[:-1]
        inputs.pop()

    '''
    if inputs[0] > inputs[len(inputs)-1]:
        updated_sidelength = inputs[0]
        updated_inputs = inputs[1:]
    else:
        updated_sidelength = inputs[len(inputs)-1]
        updated_inputs = inputs[:-1]
    '''

    # print 'updated inputs =', updated_inputs

    updated_inputs = inputs
    return updated_sidelength, updated_inputs


for i in range(test_cases):
    #num = int(raw_input())
    #print 'i =', i

    num = int(file.readline().split()[0])

    #inputs = map(int, raw_input().split())
    inputs = map(int, file.readline().split())

    current_sidelength = max(inputs[0], inputs[-1])
    #print 'start sidelength =', current_sidelength
    #print 'two values =', inputs[0], inputs[-1]

    if current_sidelength < max(inputs):
        print 'No'
        continue

    for j in range(len(inputs)):

        prev_inputs_len = len(inputs)
        updated_sidelength, updated_inputs = erase_one_element(current_sidelength, inputs)

        if len(updated_inputs) == 0:
            break
        if len(updated_inputs) == prev_inputs_len:
            print 'current inputs size =', len(inputs)
            print 'No'
            break

        inputs = updated_inputs
        current_sidelength = updated_sidelength

    if len(updated_inputs) == 0:
        print 'Yes'
    # print inputs

file.close()