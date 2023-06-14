# single sigmoid neuron
# scalar input, scalar ouput
# 2 paramaters : one w, one b

# opti algos : GD, MiniBatch, Momentum, NAG, AdaGrad, RMSProp, Adam 

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
        eps=1e-8, beta=0.9, beta1=0.9, beta2=0.9, # more params
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

        elif self.algo == "AdaGrad":
            v_w, v_b = 0, 0
            for i in range(epochs):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                # running sum of square of parameters
                v_w += dw**2
                v_b += db**2
                # different learning rate
                self.w -= (eta / np.sqrt(v_w) + eps) * dw
                self.b -= (eta / np.sqrt(v_b) + eps) * db
                self.append_log()

        elif self.algo == "RMSProp":
            v_w, v_b = 0, 0
            for i in range(epochs):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw += self.grad_w(x, y)
                    db += self.grad_b(x, y)
                v_w = beta * v_w + (1 - beta) * dw**2
                v_b = beta * v_b + (1 - beta) * db**2
                self.w -= (eta / np.sqrt(v_w) + eps) * dw
                self.b -= (eta / np.sqrt(v_b) + eps) * db
                self.append_log()

        elif self.algo == "Adam":
            v_w, v_b = 0, 0
            m_w, m_b = 0, 0
            num_updates = 0
            for i in range(epochs):
                dw, db = 0, 0
                for x, y in zip(X, Y):
                    dw = self.grad_w(x, y)
                    db = self.grad_b(x, y)
                    num_updates += 1

                    # for recurssion - not bias corrected terms.

                    m_w = beta1 * m_w + (1 - beta1) * dw
                    m_b = beta1 * m_b + (1 - beta1) * db

                    v_w = beta2 * v_w + (1 - beta2) * dw**2
                    v_b = beta2 * v_b + (1 - beta2) * db**2

                    # bias correction
                    m_w_c = m_w / (1 - np.power(beta1, num_updates))
                    m_b_c = m_b / (1 - np.power(beta1, num_updates))
                    v_w_c = v_w / (1 - np.power(beta2, num_updates))
                    v_b_c = v_b / (1 - np.power(beta2, num_updates))

                    # make bias correction just before updating and update with those.

                    self.w -= (eta / np.sqrt(v_w_c) + eps) * m_w_c
                    self.b -= (eta / np.sqrt(v_b_c) + eps) * m_b_c
                    self.append_log()



    def append_log(self):
        # append current weight, bias, error with that weight, bias
        self.W_h.append(self.w)
        self.B_h.append(self.b)
        self.E_h.append(self.error(self.X, self.Y))
