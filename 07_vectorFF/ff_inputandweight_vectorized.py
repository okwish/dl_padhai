# Feedfoward class
# inputs also also vectorized
# but this is not generic - fixed input-size, output-size, number of layer, number of neurons in each

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, log_loss

from tqdm import notebook


class FFInputWeightVectorised:
    def __init__(self, Wmat1, Wmat2):
        self.Wmat1 = Wmat1.copy()
        self.Wmat2 = Wmat2.copy()
        self.Bvec1 = np.zeros((1, 2))
        self.Bvec2 = np.zeros((1, 4))

    def sigmoid(self, AA):
        return 1.0 / (1.0 + np.exp(-AA))

    def softmax(self, AA):
        EE = np.exp(AA)
        return EE / np.sum(EE, axis=1).reshape(-1, 1)

    def forward_pass(self, XX): 
        # whole data matrix as input - using vectorization
        # matmul support vectorization
        self.AA1 = np.matmul(XX, self.Wmat1) + self.Bvec1  # (N, 2) * (2, 2) -> (N, 2)
        self.HH1 = self.sigmoid(self.AA1)  # (N, 2)
        self.AA2 = np.matmul(self.HH1, self.Wmat2) + self.Bvec2  # (N, 2) * (2, 4) -> (N, 4)
        self.HH2 = self.softmax(self.AA2)  # (N, 4)
        return self.HH2

    def grad_sigmoid(self, AA):
        return AA * (1 - AA)

    def grad(self, XX, YY):
        self.forward_pass(XX)
        m = XX.shape[0]

        self.dAA2 = self.HH2 - YY  # (N, 4) - (N, 4) -> (N, 4)

        self.dWmat2 = np.matmul(self.HH1.T, self.dAA2)  # (2, N) * (N, 4) -> (2, 4)
        self.dBvec2 = np.sum(self.dAA2, axis=0).reshape(1, -1)  # (N, 4) -> (1, 4)
        self.dHH1 = np.matmul(self.dAA2, self.Wmat2.T)  # (N, 4) * (4, 2) -> (N, 2)
        self.dAA1 = np.multiply(self.dHH1, self.grad_sigmoid(self.HH1))  # (N, 2) .* (N, 2) -> (N, 2)

        self.dWmat1 = np.matmul(XX.T, self.dAA1)  # (2, N) * (N, 2) -> (2, 2)
        self.dBvec1 = np.sum(self.dAA1, axis=0).reshape(1, -1)  # (N, 2) -> (1, 2)

    def fit(self, XX, YY, epochs=1, learning_rate=1, display_loss=False):

        if display_loss:
            loss = {}

        for i in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):
            # no loop now
            self.grad(XX, YY)  # XX -> (N, 2), YY -> (N, 4)
            # calling grad once in an epoch.. nice abstraction also.

            m = XX.shape[0]
            self.Wmat2 -= learning_rate * (self.dWmat2 / m)
            self.Bvec2 -= learning_rate * (self.dBvec2 / m)
            self.Wmat1 -= learning_rate * (self.dWmat1 / m)
            self.Bvec1 -= learning_rate * (self.dBvec1 / m)

            if display_loss:
                YY_pred = self.predict(XX)
                loss[i] = log_loss(np.argmax(YY, axis=1), YY_pred)

        if display_loss:
            plt.plot(loss.values())
            plt.xlabel("Epochs")
            plt.ylabel("Log Loss")
            plt.show()

    def predict(self, XX):
        YY_pred = self.forward_pass(XX)
        return np.array(YY_pred).squeeze()
