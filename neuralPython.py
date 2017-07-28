import numpy as np

def activationFunction(answer):
        if answer[0] < 0:
            return 0
        elif answer[0] > 0:
            return 1

inp = np.array([[1,0,0],
                   [1,0,1],
                   [1,1,0],
                   [1,1,1]])
outp = np.array([0,0,0,1])
weight = np.array([[0.1],
                   [0.1],
                   [0.1]])

learningRate = 0.5

for rnd in range(10):
    print('EPoch ', rnd,'===============')
    for innr in range(4):
        print('round ' , innr, '--------')
        result = inp[innr].dot(weight)
        print('Result = ', result[0])

        target = activationFunction(result)

        print('Target = ', target)
        print('Output = ', outp[innr])
        diffWeight = inp[innr].dot(learningRate).dot(target-outp[innr]).reshape(3,1)
        print('diffWeight is ',diffWeight)
        weight -= diffWeight
        print('weight is ', weight)

print('==============================')
print('==============================')
print('Answer is ', activationFunction(np.dot(inp[3],weight)))
