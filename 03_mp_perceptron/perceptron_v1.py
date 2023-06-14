# Perceptron class

import numpy as np

class PerceptronV1:
    
    def __init__(self):
        self.W = None #weights array
        self.b = None #scalar(bias)
        
    def model(self, X):
        return 1 if ( np.dot(self.W,X) >= self.b) else 0
        
    def predict(self, XX):
        Y=[]
        for X in XX:
            result = self.model(X)
            Y.append(result)
        return np.array(Y)
        
    def fit(self, XX, Y):
        
        #initialize parameters
        self.W = np.ones(XX.shape[1])
        self.b = 0
        
        #go through each x and modify if needed
        
        for X,y in zip(XX,Y): 
            y_pred = self.model(X)
            if y == 1 and y_pred == 0: #not matching. need update
                self.W = self.W + X
                self.b = self.b - 1
                #comes from vector thing
            elif y == 0 and y_pred == 1:
                self.W = self.W - X
                self.b = self.b + 1
            #not making any change if they agree(y and y_pred).
