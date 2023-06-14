# Perceptron class that populates and returns a weight matrix
# weight after each epochs - list

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class PerceptronV5:
    
    def __init__(self):
        self.W = None #weights
        self.b = None #bias
        
    def model(self, X):
        return 1 if ( np.dot(self.W,X) >= self.b) else 0
        
    def predict(self, XX):
        Y=[]
        for X in XX:
            result = self.model(X)
            Y.append(result)
        return np.array(Y)
        
    def fit(self, XX, Y, epochs = 1, lr =1):
        
        #initialize parameters
        self.W = np.ones(XX.shape[1])
        self.b = 0
        
        accuracy = {} #accuracy after each epoch
        max_accuracy = 0
        
        wt_matrix = []        
        
        for i in range(epochs):
            for X,y in zip(XX,Y):
                y_pred = self.model(X)
                if y == 1 and y_pred == 0: #not matching. need update
                    self.W = self.W + lr*X #learning rate also
                    self.b = self.b - lr*1
                    #comes from vector thing
                elif y == 0 and y_pred == 1:
                    self.W = self.W - lr*X
                    self.b = self.b + lr*1
                #not making any change if they(y and Y_pred) agree.
                
            # as we iterate through epochs, populate weights after each into the weight matrix
            # finally return the weight matrix
            wt_matrix.append(self.W) #matrix with rows as weights-after-each-epoch
                
            accuracy[i] = accuracy_score(self.predict(XX),Y)
            if(accuracy[i]>max_accuracy):
                max_accuracy = accuracy[i]
                #checkpointing
                chkpt_W = self.W
                chkpt_b = self.b
        
        self.W = chkpt_W
        self.b = chkpt_b
        #after all epochs set W, b to best parameters
        
        print(max_accuracy)      
        plt.plot(accuracy.values())
        plt.ylim([0,1]) #accuracy will be b/w 0,1 - to see the variation well
        plt.show()
        
        return np.array(wt_matrix)