# Generic multiple-ouptut FF network class
# given : number of inputs, number of layers, number of neurons in each layer
# given : number of outputs

# sigmoid neurons
# softmax-on-preactivation in ouptut layer

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from tqdm import notebook


class MultiClassFFNet:

    def __init__(self, n_inputs, n_outputs, hidden_sizes=[3]):
        self.nx = n_inputs
        self.ny = n_outputs
        self.nh = len(hidden_sizes) #number of hidden layers
        self.sizes = [self.nx] + hidden_sizes + [self.ny]

        # PARAMETERS

        self.Wmat_dict = {}
        self.Bvec_dict = {}
        for i in range(self.nh + 1):
            self.Wmat_dict[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
            self.Bvec_dict[i + 1] = np.zeros((1, self.sizes[i + 1]))

    # FORWARD PASS

    def sigmoid(self, A):
        return 1.0 / (1.0 + np.exp(-A))

    def softmax(self, A):
        E = np.exp(A) # exponentials
        return E / np.sum(E)

    def forward_pass(self, X):
        self.Avec_dict = {}
        self.Hvec_dict = {}
        self.Hvec_dict[0] = X.reshape(1, -1)
        # hidden layers
        for i in range(self.nh):
            self.Avec_dict[i + 1] = np.matmul(self.Hvec_dict[i], self.Wmat_dict[i + 1]) + self.Bvec_dict[i + 1]
            self.Hvec_dict[i + 1] = self.sigmoid(self.Avec_dict[i + 1])
        # output layer
        self.Avec_dict[self.nh + 1] = (
            np.matmul(self.Hvec_dict[self.nh], self.Wmat_dict[self.nh + 1]) + self.Bvec_dict[self.nh + 1]
        )
        self.Hvec_dict[self.nh + 1] = self.softmax(self.Avec_dict[self.nh + 1]) # softmax activation
        return self.Hvec_dict[self.nh + 1]

    # PREDICT

    def predict(self, XX):
        YY_pred = []
        for X in XX:
            Y_pred = self.forward_pass(X) #Y_pred will be a vector
            YY_pred.append(Y_pred)
        return np.array(YY_pred).squeeze()
        # squeeze removes the axes-of-length-1

    # GRADIENT

    def grad_sigmoid(self, X):
        return X * (1 - X)

    def grad(self, X, Y):
        self.forward_pass(X)
        self.dWmat_dict = {}
        self.dBvec_dict = {}
        self.dHvec_dict = {}
        self.dAvec_dict = {}
        L = self.nh + 1
        self.dAvec_dict[L] = self.Hvec_dict[L] - Y
        for k in range(L, 0, -1):
            self.dWmat_dict[k] = np.matmul(self.Hvec_dict[k - 1].T, self.dAvec_dict[k])
            self.dBvec_dict[k] = self.dAvec_dict[k]
            self.dHvec_dict[k - 1] = np.matmul(self.dAvec_dict[k], self.Wmat_dict[k].T)
            self.dAvec_dict[k - 1] = np.multiply(self.dHvec_dict[k - 1], self.grad_sigmoid(self.Hvec_dict[k - 1]))


    # LOSS

    def cross_entropy(self, YY, YY_pred): # ground truth, prediction
        LLmat = np.multiply(YY_pred, YY) # element wise multiplication
        # only those corresponding to "hot-1s" will remain non-zero. 
        Lvec = LLmat[LLmat != 0] # return array. with those elements
        L = -np.log(Lvec)
        l = np.mean(L)
        return l

    # FIT
     
    def fit(self, XX, YY, epochs=100, initialize="True", learning_rate=0.01, display_loss=False):
        if display_loss:
            loss = {}

        if initialize:
            for i in range(self.nh + 1):
                self.Wmat_dict[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
                self.Bvec_dict[i + 1] = np.zeros((1, self.sizes[i + 1]))

        for epoch in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):
            dW = {} # for summing gradient over datapoints
            dB = {}
            for i in range(self.nh + 1):
                dW[i + 1] = np.zeros((self.sizes[i], self.sizes[i + 1]))
                dB[i + 1] = np.zeros((1, self.sizes[i + 1]))
            for X, Y in zip(XX, YY):
                self.grad(X, Y)
                for i in range(self.nh + 1):
                    dW[i + 1] += self.dWmat_dict[i + 1]
                    dB[i + 1] += self.dBvec_dict[i + 1]

            m = XX.shape[0]
            for i in range(self.nh + 1):
                self.Wmat_dict[i + 1] -= learning_rate * (dW[i + 1] / m)
                self.Bvec_dict[i + 1] -= learning_rate * (dB[i + 1] / m)

            if display_loss:
                YY_pred = self.predict(XX)
                loss[epoch] = self.cross_entropy(YY, YY_pred)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel("Epochs")
            plt.ylabel("CE")
            plt.show()
