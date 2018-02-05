from layer import *
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle


'''
Dropout Is Used To Force Neurons To Make More Pathways And Thererfore Deal Better With
Unexpected New Data

L1 Regularization Is Used To Decrease Model Complexity, For Example Instead Of Using A
Polynomial Of High Order It Makes The Network Learn A Model Of Lower Order

L2 Regularization Also Decreases Model Complexity But Also Increases Sparsity Thererfore
Neurons Which Would Have Normally Fired Will Now Stop
'''




class Network:


    #
    # Constructor Methods
    #

    def __init__(self, shape = None, fileName = None):
        # If A Shape Is Defined Then Create Network Based On That
        if not shape == None:
            self.__createNetworkBasedOnShape(shape)
        # If A File Path Is Specified Load It
        elif not fileName == None:
            self.__loadModel(fileName)
        # If None Of Them Are Given Then The Network Will Be Defined Seperatly
        else:
            # Create Only The Container For The Layers
            self.layers = []


    def __createNetworkBasedOnShape(self, shape):
        self.layers = []
        self.layers.append(Layer([shape[0]]))
        for num, numPrevLayer in zip(shape[1:], shape[:-1]):
            self.layers.append(Layer([num, numPrevLayer], Sigmoid))


    def createLayer(self, neurons, activationFunction = Sigmoid, dropoutKeepProbability = 1):
        # Add Layer Based On Given Information
        self.layers.append(Layer(neurons, activationFunction, dropoutKeepProbability))


    #
    # Prediction Methods
    #

    def predict(self, x):
        # Reshape Row To A Colomn
        x = np.array(x).reshape(-1, 1)
        # Give The Input Layer The Input
        self.layers[0].setA(x)
        # Feed Through Network
        for layerNum in range(1, len(self.layers)):
            self.layers[layerNum].feedForward(self.layers[layerNum-1].getA())
        # Return The Activity Of The Last Layer
        return self.layers[-1].getA()


    #
    # Training Methods
    #

    def quickFit(self, inputData, targetData, checkAfter = 500, targetError = 0.01, learningRate = 0.005, momentum = 0.7, logging = False):
        '''  Just Loops Until Target Error Is Reached   '''

        # User Notification
        print("Start Training")

        # Prepare For Logging If Wanted
        if logging:
            self.__prepareForLogging()
        # Preapare All Layers For Training
        self.__prepareForTraining(momentum)

        # Varibale Holding The Error
        error = 10
        # Varibale Holding The Amount Of Times Looped Through All The Training Data
        totalIterations = 0
        # Varibale Controlling If The Training Is Finished
        targetErrorReacherd = False

        # Do Is For As Long The Error Is To Big
        while not targetErrorReacherd:
            # Go Trough All Examples And Perform Gradient Decent
            for index, (x, y) in enumerate(zip(inputData, targetData)):
                # Let Network Train On That One Example
                error = self.__trainOneExample(x, y, learningRate)

                # After Every ...batchSize
                if index % checkAfter == 0:
                    # User Notification
                    print(error)
                    # If Logging Is Wanted Add An Entry In The Error Log
                    if logging:
                        self.errorLog.append([totalIterations * len(inputData) + index, error])
                    # Is Target Error Reached
                    if error <= targetError:
                        # Mark That The Target Error Is Reached
                        targetErrorReacherd = True
                        # And Break Out Of The Current Training Loop
                        break
            # User Notification
            print("Error Of", error)
            # Add One To The Number Of Total Iterations
            totalIterations = totalIterations + 1

        # Cleanup Variabels
        self.__cleanUp()

        # User Notification
        print("Finished Training With Error Of", error, "After", (totalIterations-1) * len(inputData) + index)



    def maxFit(self, inputData, targetData, targetError = 0.1, learningRate = 0.005, momentum = 0.7, logging = False):
        ''' Loops Until The Average Eror Over All Training Data Is At Target Error '''

        # User Notification
        print("Start Training")

        # Prepare For Logging If Wanted
        if logging:
            self.__prepareForLogging()
        # Preapare All Layers For Training
        self.__prepareForTraining(momentum)

        # Varibale Holding The Error
        error = 10
        # Varibale Holding The Amount Of Times Looped Through All The Training Data
        totalIterations = 0
        # Varibale Controlling If The Training Is Finished
        targetErrorReacherd = False

        # Do Is For As Long The Error Is To Big
        while not targetErrorReacherd:
            # Variabel Holding The Error Sum Over All The Trainig Data
            errorSum = 0

            # Go Trough All Examples And Perform Gradient Decent
            for index, (x, y) in enumerate(zip(inputData, targetData)):
                # Let Network Train On That One Example
                error = self.__trainOneExample(x, y, learningRate)
                # Add The Error To The Error Sum
                errorSum += error


            # Calulate The Average Error Over Alll The Training Data
            averageError = errorSum / len(inputData)
            # User Notification
            print("In Iteration", totalIterations+1, "There Is An Average Error Over All Training Data Of:", averageError)

            # Check If The Average Error Is Smaller Than The Target Error
            if averageError <= targetError:
                # Mark That The Target Error Got Reached
                targetErrorReacherd = True

            # If Logging Is Wanted Add An Entry In The Error Log
            if logging:
                self.errorLog.append([totalIterations * len(inputData) + index, averageError])

            # Add One To The Number Of Total Iterations
            totalIterations = totalIterations + 1


        # Cleanup Variabels
        self.__cleanUp()

        # User Notification
        print("Finished Training With Error Of", error, "After", (totalIterations-1) * len(inputData) + index)


    def __trainOneExample(self, x, y, learningRate):
        # Reshape Row To A Column
        y = np.array(y).reshape(-1, 1)
        # Make A Prediction
        prediction = self.predict(x)
        # Calculate Cost On Output layer
        cost = prediction - y
        # Calculate Total Network Cost
        totalCost = 0.5 * np.sum(cost**2)

        # Create A Container Holding The Errors
        errors = []
        # Calculate The Error On The Output Layer And Add It To The Container
        errorOutputLayer = np.multiply(self.layers[-1].activation(self.layers[-1].getZ(), derivative = True), cost)
        errors.append(errorOutputLayer)
        # Calculate The Errors For The Hidden Layers
        for layerNum in range(len(self.layers)-2, 0, -1):
            # Calulate Weighted Error Of The Next Layer
            weightedErrorNextLayer = np.dot(self.layers[layerNum+1].getW().T, errors[0])
            # Insert The Error Of That Layer Into The Container
            errors.insert(0, np.multiply(weightedErrorNextLayer, self.layers[layerNum].activation(self.layers[layerNum].getZ(), derivative = True)))
        # Insert Placeholder To Equal Out Layersize and Errorsize
        errors.insert(0, 0)

        # Update All The Weights And Biases
        for layerNum in range(1, len(self.layers)):
            self.layers[layerNum].updateWeightsAndBias(errors[layerNum], self.layers[layerNum-1].getA(), learningRate)

        # Return The Total Cost Of The Network
        return totalCost


    #
    # Helper Methods
    #

    def saveModel(self, fileName):
        # Save Model Using Pickle Library
        with open(fileName + '.pkl', 'wb') as writer:
            pickle.dump(self.layers, writer)

    def showErrorLog(self):
        example, error = zip(*self.errorLog)
        plt.plot(example, error)
        plt.xlabel('Example Count')
        plt.ylabel('Error')
        plt.show()

    def getLogginData(self):
        return self.errorLog

    #
    # Private Methods
    #

    def __prepareForLogging(self):
        # Create Variabels Such As The Error Log
        self.errorLog = []

    def __prepareForTraining(self, momentum):
        for layerNum in range(1, len(self.layers)):
            self.layers[layerNum].preapareForTraining(momentum)

    def __cleanUp(self):
        for layerNum in range(1, len(self.layers)):
            self.layers[layerNum].cleanup()

    def __loadModel(self, fileName):
        # Load Model Using Pickle Library
        with open(fileName + '.pkl', 'rb') as reader:
            self.layers = pickle.load(reader)












    def log(self, x, y, prediction, cost, totalCost, errors, learningRate):
        print("\n-----------------\n\n", "# Network Cost:")
        print("Prediction:\n", prediction, "\nExpected:\n", y, "\nCost:\n", cost, "\nTotalCost:", totalCost)
        print("\n\n# Errors:")
        for index, e in enumerate(errors):
            print("Layer", index+1, "Has Error:\n", e)
        print("\n\n# Deltas:")
        for layerNum in range(1, len(self.layers)):
            print("Bias Delta:\n", learningRate * errors[layerNum])
            print("\nWeight Detla:\n", learningRate * np.dot(errors[layerNum], self.layers[layerNum-1].getA().T))
        print("\n-----------------\n")
