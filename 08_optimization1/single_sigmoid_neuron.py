# single sigmoid neuron
# scalar input, scalar ouput
# 2 paramaters : one w, one b

# opti algos : GD, MiniBatch, Momentum, NAG

import numpy as np


class SingleSigmoidNeuron:

    def __init__(self, w_init, b_init, algo): 
        # passing specific init values, so that we can visualise for different intial values
        # algo - which optimization algorithm to use

        # PARAMETERS

        self.w = w_init
        self.b = b_init

        # history variables
        # track all past weights, biases, error
        self.W_h = []
        self.B_h = []
        self.E_h = []
        
        self.algo = algo

    # FORWARD 

    def sigmoid(self, a, w=None, b=None):
        # generally we use self.w and self.b in sigmoid (ie, current w,b values)
        # but there is a need to run sigmoid at other w,b also.
        # eg: in nesterov we find gradient at another w,b
        # therefore an option to find sigmoid at a specified w, b also.
        # the same thing in grad,etc. also.
        if w is None:
            w = self.w
        if b is None:
            b = self.b

        return 1.0 / (1.0 + np.exp(-(w * a + b)))

    # ERROR
    
    # mean square error
    def error(self, X, Y, w=None, b=None): 
        if w is None:
            w = self.w
        if b is None:
            b = self.b

        e = 0 #error
        for x, y in zip(X, Y): # sum over datapoints
            e += 0.5 * (self.sigmoid(x, w, b) - y) ** 2
        return e

    # GRADIENT

    def grad_w(self, x, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b

        y_pred = self.sigmoid(x, w, b)
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    def grad_b(self, x, y, w=None, b=None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        y_pred = self.sigmoid(x, w, b)
        return (y_pred - y) * y_pred * (1 - y_pred)

    # FIT
    
    def fit( self,
        X, Y, epochs=100,
        eta=0.01, gamma=0.9, mini_batch_size=100,
    ):
        self.W_h = []
        self.B_h = []
        self.E_h = []
        self.X = X
        self.Y = Y

        # different algorithms

        if self.algo == "GD":
            for i in range(epochs):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                self.w -= eta * dw / X.shape[0]
                self.b -= eta * db / X.shape[0]
                # log everytime update is made
                self.append_log()

        elif self.algo == "MiniBatch":
            # summing gradient over a minibatch of datapoints only
            for i in range(epochs):
                dw, db = 0, 0
                points_seen = 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                    points_seen += 1  # counting datapoints
                    if points_seen % mini_batch_size == 0:                       
                        self.w -= eta * dw / mini_batch_size
                        self.b -= eta * db / mini_batch_size
                        # dividing by mini batch size (average) - not full data size.
                        self.append_log()
                        dw, db = 0, 0

        elif self.algo == "Momentum":
            v_w, v_b = 0, 0  # will have the past v values - for the next epoch.
            for i in range(epochs):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                v_w = gamma * v_w + eta * dw
                v_b = gamma * v_b + eta * db
                self.w = self.w - v_w
                self.b = self.b - v_b
                self.append_log()

        elif self.algo == "NAG":
            v_w, v_b = 0, 0
            for i in range(epochs):
                dw, db = 0, 0
                v_w = gamma * v_w
                v_b = gamma * v_b
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y, self.w - v_w, self.b - v_b)
                    db += self.grad_b(x, y, self.w - v_w, self.b - v_b)
                    # gradients at different values
                v_w = v_w + eta * dw
                v_b = v_b + eta * db
                self.w = self.w - v_w
                self.b = self.b - v_b
                self.append_log()

    def append_log(self):
        # append current weight, bias, error with that weight, bias
        self.W_h.append(self.w)
        self.B_h.append(self.b)
        self.E_h.append(self.error(self.X, self.Y))
