#Sigmoid neuron class 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from tqdm import notebook

# general implementation - multiple inputs 

class SigmoidNeuronV3:

    def __init__(self):
        self.W = None 
        self.b = None 
        
    # splitting model into a linear pass(pre-activation) (weighted combination of inputs)
    # and then a sigmoid on that value(activation)
    def linear(self, X):
        return np.dot(self.W,X)+self.b

    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def forward(self, X):
        return self.sigmoid(self.linear(X))
    
    # gradient of loss wrt parameters
    # W part of gradient
    def grad_W(self, X, y):
        #forward pass - (as its needed in the gradient formula)
        fx = self.forward(X) #y_pred "f(Xi)"

        #gradient:
        return (fx-y)*fx*(1-fx)*X
    
    # b part of gradient
    def grad_b(self, X, y):
        #forward pass - (as its needed in the gradient formula)
        fx = fx = self.forward(X) #y_pred "f(Xi)"

        #gradient:
        return (fx-y)*fx*(1-fx)
    
        
    def fit(self, XX, Y, epochs = 1, alpha = 1.0 , initialize = True, display_loss=False): 
        # find the optimal parameters
        # learning algo - gradient descent
        
        # initialize parameters
        if initialize: #sometimes we might not want to initialize
            # eg. already ran once and want to use the parameters learned later
            self.W = np.random.randn(1,XX.shape[1])
            self.b = 0
        
        if display_loss:
            loss = {}

        # for i in range(epochs):
        for i in notebook.tqdm( range(epochs), total=epochs, unit="epochs" ):
            dW,db = 0,0
            # gradient 
            # sum over all datapoints
            for X,y in zip(XX,Y): # iterate over datapoints
                dW += self.grad_W(X,y)
                db += self.grad_b(X,y)
                
            # update W, b (each epoch)
            self.W -= alpha*dW
            self.b -= alpha*db

            # populate loss after each epoch
            if display_loss:
                Y_pred = self.predict(XX) #prediction on data with current parameters
                loss[i] = mean_squared_error(Y_pred,Y)
        
        # after training(after all epochs) - plot the populated loss-after-each-epoch
        if display_loss:
            plt.plot(loss.values())
            plt.xlabel('epochs')
            plt.ylabel('MSE')
            plt.show()

    def predict(self, XX):
        #find ouput vector for a given input 'data' - with current parameters
        # do prediction on data
        Y_pred = []
        for X in XX:
            y_pred = self.forward(X)
            Y_pred.append(y_pred)
        return np.array(Y_pred)     