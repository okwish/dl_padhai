# Generic FF network class
# given : number of inputs, number of layers, number of neurons in each layer
# one output

# sigmoid neurons

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from tqdm import notebook


class GenericFFNet:
    def __init__(self, n_inputs, hidden_sizes=[2]):
        # hidden_sizes -> list with number of neurons in each layer.
        # len of list = number of hidden layers.

        self.nx = n_inputs
        self.ny = 1  # one output
        self.nh = len(hidden_sizes)  # number of hidden layers
        self.sizes = [self.nx] + hidden_sizes + [self.ny]  # concatenate
        # list with numbers in all layers (input - hidden - ouput)

        # PARAMETERS - these are going to be optimized(learned)

        self.Wmat_dict = {}  # dictionary where each item is a weight matrix.
        # Wmat_dict[i] is a matrix
        self.Bvec_dict = {}  # dictionary where each item is a bias vector.

        # initialise weights to random values and biases to 0
        for i in range(self.nh + 1):
            # dimensions using the sizes list.
            self.Wmat_dict[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
            self.Bvec_dict[i + 1] = np.zeros((1, self.sizes[i + 1]))

    # FORWARD PASS

    def sigmoid(self, A):
        return 1.0 / (1.0 + np.exp(-A))

    def forward_pass(self, X):
        self.Avec_dict = {}  # dictionary where each item is a-vector(preactivation)
        self.Hvec_dict = {}  # dictionary where each item is a-vector
        self.Hvec_dict[0] = X.reshape(1, -1)  # taking H[0] as input.
        # reshaping to make sure it has one row(ie, its a vector).. (input can be image.)

        for i in range(self.nh + 1):
            # computing 'a' first, then 'h'
            self.Avec_dict[i + 1] = (
                np.matmul(self.Hvec_dict[i], self.Wmat_dict[i + 1]) + self.Bvec_dict[i + 1]
            )
            self.Hvec_dict[i + 1] = self.sigmoid(self.Avec_dict[i + 1])

        return self.Hvec_dict[self.nh + 1]  # H - final layer is the y value.


    # GRADIENT

    def grad_sigmoid(self, X):
        return X * (1 - X)

    def grad(self, X, y):
        self.forward_pass(X)
        self.dWmat_dict = {}
        self.dBvec_dict = {}
        self.dHvec_dict = {}
        self.dAvec_dict = {}
        L = self.nh + 1
        self.dAvec_dict[L] = self.Hvec_dict[L] - y
        # coming layer by layer from back
        # backpropagation
        for k in range(L, 0, -1):
            self.dWmat_dict[k] = np.matmul(self.Hvec_dict[k - 1].T, self.dAvec_dict[k])
            self.dBvec_dict[k] = self.dAvec_dict[k]
            self.dHvec_dict[k - 1] = np.matmul(self.dAvec_dict[k], self.Wmat_dict[k].T)
            self.dAvec_dict[k - 1] = np.multiply(
                self.dHvec_dict[k - 1], self.grad_sigmoid(self.Hvec_dict[k - 1])
            )

    # FIT

    def fit(self, XX, Y, epochs=1, learning_rate=1, initialise=True, display_loss=False):

        # initialise
        if initialise:
            for i in range(self.nh + 1):
                self.Wmat_dict[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
                self.Bvec_dict[i + 1] = np.zeros((1, self.sizes[i + 1]))

        if display_loss:
            loss = {}

        for e in notebook.tqdm(range(epochs), total=epochs, unit="epoch"): #e-th epoch
            # dictionary for summing across datapoints
            dW = {}  # dictionary of dW in each layer.
            dB = {}
            for i in range(self.nh + 1):
                dW[i + 1] = np.zeros((self.sizes[i], self.sizes[i + 1]))
                dB[i + 1] = np.zeros((1, self.sizes[i + 1]))

            for X, y in zip(XX, Y):
                self.grad(X, y) # find grad
                # sum
                for i in range(self.nh + 1):
                    dW[i + 1] += self.dWmat_dict[i + 1]
                    dB[i + 1] += self.dBvec_dict[i + 1]

            m = XX.shape[0]
            for i in range(self.nh + 1):
                self.Wmat_dict[i + 1] -= learning_rate * dW[i + 1] / m
                self.Bvec_dict[i + 1] -= learning_rate * dB[i + 1] / m

            if display_loss:
                Y_pred = self.predict(XX)
                loss[e] = mean_squared_error(Y_pred, Y)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel("Epochs")
            plt.ylabel("Mean Squared Error")
            plt.show()

    def predict(self, XX):
        Y_pred = []
        for X in XX:
            # forward pass on a set of inputs given
            y_pred = self.forward_pass(X)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()
