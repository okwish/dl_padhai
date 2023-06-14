#MP neuron implementation as a class.
#general template for implementing all models (as class)

import numpy as np
from sklearn.metrics import accuracy_score

class MPNeuron:
    
    def __init__(self):
        #define parameters of model
        self.b = None #initialise as none
    
    def model(self, X):
        #fn that takes one input X and predit its ouput y
        #one input x. - model is that only
        return(sum(X) >= self.b )
    
    def predict(self, XX):
        #predict ouput for entire data XX
        Y = []
        for X in XX:
            result = self.model(X)
            Y.append(result)
        return np.array(Y)
    
    def fit(self, XX, Y):
        #learning algorithm to find the right value of parameters
        accuracy = {} #dict for accuracy for different parameter values.
        # accuracy is only a local variable. 
        # no need to make it a class attribute as is not need by other class functions.
        for b in range(XX.shape[1]+1): # b range from 0 to #datapoints
            self.b = b #as model functions (predict,etc.) use self.b
            Y_pred = self.predict(XX) 
            accuracy[b] = accuracy_score(Y_pred, Y) #imported sklearn fn; global function

        # accuracy dict - b:accuracy_score
        # max(dict) => max among "values". but we need key of max value.    
        best_b = max(accuracy, key=accuracy.get) #return key with the max 'value'
        # get -> fn that return key of item with that value in a dictionary
        # max(.... , key = fn)- return the result of applying "that function"(key argument) on the original max ouput
                
        self.b = best_b # update optimal b in the model itlsef (self parameter)
        
        print('optimal b is:', best_b)
        print('highest acc:',accuracy[best_b])  
        