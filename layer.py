import numpy as np
from transferfunction import *



class Layer:

    def __init__(self, neurons, activationFunction, dropoutKeepProbability):
        self.activation = activationFunction
        self.dropoutKeepProbability = dropoutKeepProbability
        self.a = 0
        self.z = 0
        if len(neurons) > 1:
            self.w = np.random.uniform(low=-0.3, high=0.3, size=(neurons[0], neurons[1]))
            self.b = np.full((neurons[0], 1), 0.0)


    def feedForward(self, aPrevLayer):
        # Sum Together And Add Bias
        self.z = np.dot(self.w, aPrevLayer) + self.b
        # Activate
        self.a = self.activation(self.z)
        # Apply Dropout Mask
        self.a = self.a * np.random.binomial(1, self.dropoutKeepProbability, size = self.a.shape)

    def updateWeightsAndBias(self, error, aPrevLayer, learningRate):
        # Update Bias
        self.b -= learningRate * error
        # Calulate The Delta For The Weights
        deltaW = learningRate * np.dot(error, aPrevLayer.T)
        # Update Weights
        self.w -= deltaW + ( self.momentum * self.prevDeltaW )
        # Remember The Delta For Next Run
        self.prevDeltaW = deltaW



    def preapareForTraining(self, momentum):
        ''' Used To Create Necessary Variabels (Momentum) '''
        self.prevDeltaW = np.full(self.w.shape, 0.0)
        self.momentum = momentum

    def cleanup(self):
        ''' Used To Delete Trainig Specific Variabels (Momentum) '''
        del self.prevDeltaW
        del self.momentum



    def getA(self):
        return self.a
    def getZ(self):
        return self.z
    def getW(self):
        return self.w
    def setA(self, x):
        self.a = x
