import numpy as np
import math



def clip(x, minVal, maxVal):
    if x < minVal:
        return minVal
    elif x > maxVal:
        return maxVal
    else:
        return x

def __sig(x):
    x = clip(x, -500, 500)
    return 1 / (1 + math.exp(-x))


def Softmax(x, derivative = False):
    if not derivative:
        expSum = 0
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                exp = math.exp(x[row][col])
                x[row][col] = exp
                expSum += exp
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                x[row][col] = x[row][col] / expSum
    else:
        x = x * (1 - x)
    return x


def ReLu(x, derivative = False):
    if not derivative:
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                x[row][col] = max(0, x[row][col])
    else:
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                x[row][col] = 1 if x[row][col] > 0 else 0
    return x

def Tanh(x, derivative = False):
    if not derivative:
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                x[row][col] = math.tanh(x[row][col])
    else:
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                x[row][col] = 1 - (math.tanh(x[row][col])**2)
    return x


def Sigmoid(x, derivative = False):
    if not derivative:
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                x[row][col] = __sig(x[row][col])
    else:
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                s = __sig(x[row][col])
                x[row][col] = s * (1 - s)
    return x
