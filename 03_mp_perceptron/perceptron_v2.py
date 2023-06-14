# Perceptron class
# Go through data 'epoch' number of times and update parameters

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class PerceptronV2:
    
    def __init__(self):
        self.W = None #weights
        self.b = None #bias
        
    def model(self, X):
        return 1 if (np.dot(self.W,X)>=self.b) else 0
        
    def predict(self, XX):
        Y=[]
        for X in XX:
            result = self.model(X)
            Y.append(result)
        return np.array(Y)
        
    def fit(self, XX, Y, epochs = 1):
        
        #initialize parameters
        self.W = np.ones(XX.shape[1])
        self.b = 0
        
        accuracy = {} #accuracy after each epoch
        max_accuracy = 0
        
        for i in range(epochs):
            for X,y in zip(XX,Y):
                y_pred = self.model(X)
                if y == 1 and y_pred == 0: #not matching. need update
                    self.W = self.W + X
                    self.b = self.b - 1
                    #comes from vector thing
                elif y == 0 and y_pred == 1:
                    self.W = self.W - X
                    self.b = self.b + 1
                #not making any change if they(y and y_pred) agree.
                
            accuracy[i] = accuracy_score(self.predict(XX),Y) # accuracy after i-th epoch
            if(accuracy[i]>max_accuracy):
                max_accuracy = accuracy[i]
             
        print(max_accuracy)      
        plt.plot(accuracy.values())
        plt.show()

        