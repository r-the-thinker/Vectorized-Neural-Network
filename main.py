from network import *
from mnistreader import *
import numpy as np
import matplotlib.pyplot as plt

# This Works
'''
tX = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
tY = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]

net = Network((2, 10, 2))
error = 0
for iters in range(10000):
    for x, y in zip(tX, tY):
        error = net.train(x,y, 1.2)
    print(error)


print(net.predict(tX[0]))
print(net.predict(tX[1]))
print(net.predict(tX[2]))
print(net.predict(tX[3]))
'''

def map0to1(val, valMax):
    return val/valMax



# Load Trainigs Data
rawImages, rawLabels, numImagePixels = get_data_and_labels("C:\\Users\\Robin\\Documents\\MNIST\\Images\\train-images.idx3-ubyte", "C:\\Users\\Robin\\Documents\\MNIST\\Labels\\train-labels.idx1-ubyte")

# Prepare Data
print("Start Preparing Data")
images = []
labels = []
for i in rawImages:
    insert = []
    for pixel in i:
        insert.append(map0to1(pixel, 255))
    images.append(insert)
for l in rawLabels:
    y = [0] * 10
    y[l] = 1
    labels.append(y)
print("Finished Preparing Data")
# CLEANUP
del rawImages

net = Network()
net.createLayer([784])
net.createLayer([800, 784], Sigmoid)
net.createLayer([10, 800], Softmax)

net.maxFit(images, labels, learningRate = 0.005, momentum = 0.79, targetError = 0.09, logging = True)
net.saveModel('model1')
net.showErrorLog()

print("Predicted For(", rawLabels[1], ")\n", net.predict(images[1]))


'''
# Load Test Data
rawImages, rawLabels, numImagePixels = get_data_and_labels("C:\\Users\\Robin\\Documents\\MNIST\\Images\\t10k-images.idx3-ubyte", "C:\\Users\\Robin\\Documents\\MNIST\\Labels\\t10k-labels.idx1-ubyte")

# Prepare Data
print("Start Preparing Data")
images = []
labels = []
for i in rawImages:
    insert = []
    for pixel in i:
        insert.append(map0to1(pixel, 255))
    images.append(insert)
for l in rawLabels:
    y = [0] * 10
    y[l] = 1
    labels.append(y)
print("Finished Preparing Data")
# CLEANUP
del rawImages


testNet = Network(fileName='model1')
print("Predicted For(", rawLabels[1], ")\n", testNet.predict(images[1]))
print("\n")
print("Predicted For(", rawLabels[53], ")\n", testNet.predict(images[53]))
print("\n")
print("Predicted For(", rawLabels[100], ")\n", testNet.predict(images[100]))
'''
