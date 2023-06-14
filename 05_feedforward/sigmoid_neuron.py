# Sigmoid neuron class

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, log_loss

from tqdm import notebook


class SigmoidNeuron:

    # PARAMETERS

    def __init__(self):
        self.W = None
        self.b = None


    # FORWARD PASS

    def linear(self, X):
        return np.dot(self.W, X) + self.b

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        return self.sigmoid(self.linear(X))


    # GRADIENT

    # loss = mean square error (for regression)
    def mse_grad_W(self, X, y):
        y_pred = self.forward(X)
        return (y_pred - y) * y_pred * (1 - y_pred) * X

    def mse_grad_b(self, X, y):
        y_pred = self.forward(X)
        return (y_pred - y) * y_pred * (1 - y_pred)

    # loss = cross entroy (for classification)
    def ce_grad_W(self, X, y):
        y_pred = self.forward(X)
        if y == 0:
            return y_pred * X
        elif y == 1:
            return -1 * (1 - y_pred) * X
        else:
            raise ValueError("y should be 0 or 1")

    def ce_grad_b(self, X, y):
        y_pred = self.forward(X)
        if y == 0:
            return y_pred
        elif y == 1:
            return -1 * (1 - y_pred)
        else:
            raise ValueError("y should be 0 or 1")


    # FIT

    def fit(
        self,
        XX,
        Y,
        epochs=1,
        learning_rate=1,
        initialise=True,
        loss_fn="mse",
        display_loss=False,
    ):
        # initialise w, b
        if initialise:
            self.W = np.random.randn(1, XX.shape[1])
            self.b = 0

        if display_loss:
            loss = {}

        for i in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):
            dw = 0
            db = 0
            for X, y in zip(XX, Y):
                if loss_fn == "mse":
                    dw += self.mse_grad_W(X, y)
                    db += self.mse_grad_b(X, y)
                elif loss_fn == "ce":
                    dw += self.ce_grad_W(X, y)
                    db += self.ce_grad_b(X, y)

            m = XX.shape[1]
            self.W -= learning_rate * dw / m
            self.b -= learning_rate * db / m
            # computing the gradient with m datapoints => divide the gradient by m.

            if display_loss:
                Y_pred = self.predict(XX)
                if loss_fn == "mse":
                    loss[i] = mean_squared_error(Y, Y_pred)
                elif loss_fn == "ce":
                    loss[i] = log_loss(Y, Y_pred)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel("Epochs")
            if loss_fn == "mse":
                plt.ylabel("Mean Squared Error")
            elif loss_fn == "ce":
                plt.ylabel("Log Loss")
            plt.show()


    # PREDICT

    def predict(self, XX):
        Y_pred = []
        for X in XX:
            y_pred = self.forward(X)
            Y_pred.append(y_pred)
        return np.array(Y_pred)
