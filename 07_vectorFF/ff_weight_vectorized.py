# Feedforward class(multiclass)
# parameters - weights matrics, bias vectors

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, log_loss

from tqdm import notebook


class FFWeightVectorised:
    def __init__(self, Wmat1, Wmat2):
        # using same weights made before
        self.Wmat1 = Wmat1.copy()
        self.Wmat2 = Wmat2.copy()
        self.Bvec1 = np.zeros((1, 2))
        self.Bvec2 = np.zeros((1, 4))

    # all as vectorised implementations..

    def sigmoid(self, A):
        # because np.exp is used - as it supports vectorisation, broadcasting
        # therefore supports a vector input
        # doing operation on each element of array and return an array of same size.(broadcasting)
        return 1.0 / (1.0 + np.exp(-A))

    def softmax(self, A):
        E = np.exp(A) #exponentials
        return E / np.sum(E)

    def forward_pass(self, X):
        # X - one input vector(arr)
        # reshape X into a column vector.
        # dimensions as comments. - keep track of the dimensions.
        X = X.reshape(1, -1)  # (1, 2)
        self.Avec1 = np.matmul(X, self.Wmat1) + self.Bvec1  # (1, 2) * (2, 2) -> (1, 2)
        self.Hvec1 = self.sigmoid(self.Avec1)  # (1, 2)
        self.Avec2 = np.matmul(self.Hvec1, self.Wmat2) + self.Bvec2  # (1, 2) * (2, 4) -> (1, 4)
        self.Hvec2 = self.softmax(self.Avec2)  # (1, 4)
        return self.Hvec2

    def grad_sigmoid(self, A):
        return A * (1 - A)

    def grad(self, X, Y):
        self.forward_pass(X)
        X = X.reshape(1, -1)  # (1, 2)
        Y = Y.reshape(1, -1)  # (1, 4)

        self.dAvec2 = self.Hvec2 - Y  # (1, 4)

        self.dWmat2 = np.matmul(self.Hvec1.T, self.dAvec2)  # (2, 1) * (1, 4) -> (2, 4)
        self.dBvec2 = self.dAvec2  # (1, 4)
        self.dHvec1 = np.matmul(self.dAvec2, self.Wmat2.T)  # (1, 4) * (4, 2) -> (1, 2)
        self.dAvec1 = np.multiply(self.dHvec1, self.grad_sigmoid(self.Hvec1))  # -> (1, 2)

        self.dWmat1 = np.matmul(X.T, self.dAvec1)  # (2, 1) * (1, 2) -> (2, 2)
        self.dBvec1 = self.dAvec1  # (1, 2)


    def fit(self, XX, YY, epochs=1, learning_rate=1, display_loss=False):
        if display_loss:
            loss = {}

        for i in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):

            # to sum gradient over datapoints
            dWmat1 = np.zeros((2, 2))
            dWmat2 = np.zeros((2, 4))
            dBvec1 = np.zeros((1, 2))
            dBvec2 = np.zeros((1, 4))

            for X, Y in zip(XX, YY):
                self.grad(X, Y)
                dWmat1 += self.dWmat1
                dWmat2 += self.dWmat2
                dBvec1 += self.dBvec1
                dBvec2 += self.dBvec2

            m = XX.shape[0]
            self.Wmat2 -= learning_rate * (dWmat2 / m)
            self.Bvec2 -= learning_rate * (dBvec2 / m)
            self.Wmat1 -= learning_rate * (dWmat1 / m)
            self.Bvec1 -= learning_rate * (dBvec1 / m)

            if display_loss:
                YY_pred = self.predict(XX)
                loss[i] = log_loss(np.argmax(YY, axis=1), YY_pred)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel("Epochs")
            plt.ylabel("Log Loss")
            plt.show()

    def predict(self, XX):
        YY_pred = []
        for X in XX:
            Y_pred = self.forward_pass(X) # input vec -> ouput vec
            YY_pred.append(Y_pred)
        return np.array(YY_pred).squeeze()
