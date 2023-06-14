# Feedforward class(multiclass) - scalar implementation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, log_loss

from tqdm import notebook


class FFScalar:

    def __init__(self, Wmat1, Wmat2):
        # initial values for weights passed
        self.w1 = Wmat1[0][0].copy()
        self.w2 = Wmat1[1][0].copy()
        self.w3 = Wmat1[0][1].copy()
        self.w4 = Wmat1[1][1].copy()
        self.w5 = Wmat2[0][0].copy()
        self.w6 = Wmat2[1][0].copy()
        self.w7 = Wmat2[0][1].copy()
        self.w8 = Wmat2[1][1].copy()
        self.w9 = Wmat2[0][2].copy()
        self.w10 = Wmat2[1][2].copy()
        self.w11 = Wmat2[0][3].copy()
        self.w12 = Wmat2[1][3].copy()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.b5 = 0
        self.b6 = 0

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def forward_pass(self, X):
        # input layer
        self.x1, self.x2 = X

        # hidden layer
        self.a1 = self.w1 * self.x1 + self.w2 * self.x2 + self.b1
        self.h1 = self.sigmoid(self.a1)
        self.a2 = self.w3 * self.x1 + self.w4 * self.x2 + self.b2
        self.h2 = self.sigmoid(self.a2)

        # output layer
        self.a3 = self.w5 * self.h1 + self.w6 * self.h2 + self.b3
        self.a4 = self.w7 * self.h1 + self.w8 * self.h2 + self.b4
        self.a5 = self.w9 * self.h1 + self.w10 * self.h2 + self.b5
        self.a6 = self.w11 * self.h1 + self.w12 * self.h2 + self.b5
        
        # softmax
        sum_exps = np.sum([np.exp(self.a3), np.exp(self.a4), np.exp(self.a5), np.exp(self.a6)])
        self.h3 = np.exp(self.a3) / sum_exps
        self.h4 = np.exp(self.a4) / sum_exps
        self.h5 = np.exp(self.a5) / sum_exps
        self.h6 = np.exp(self.a6) / sum_exps

        return np.array([self.h3, self.h4, self.h5, self.h6])
        # those are the ouput values

    def grad(self, X, Y):
        self.forward_pass(X)
        self.y1, self.y2, self.y3, self.y4 = Y # outputs

        self.da3 = self.h3 - self.y1
        self.da4 = self.h4 - self.y2
        self.da5 = self.h5 - self.y3
        self.da6 = self.h6 - self.y4

        self.dw5 = self.da3 * self.h1
        self.dw6 = self.da3 * self.h2
        self.db3 = self.da3

        self.dw7 = self.da4 * self.h1
        self.dw8 = self.da4 * self.h2
        self.db4 = self.da4

        self.dw9 = self.da5 * self.h1
        self.dw10 = self.da5 * self.h2
        self.db5 = self.da5

        self.dw11 = self.da6 * self.h1
        self.dw12 = self.da6 * self.h2
        self.db6 = self.da6

        self.dh1 = (
            self.da3 * self.w5 + self.da4 * self.w7 + self.da5 * self.w9 + self.da6 * self.w11
        )
        self.dh2 = (
            self.da3 * self.w6 + self.da4 * self.w8 + self.da5 * self.w10 + self.da6 * self.w12
        )

        self.da1 = self.dh1 * self.h1 * (1 - self.h1)
        self.da2 = self.dh2 * self.h2 * (1 - self.h2)

        self.dw1 = self.da1 * self.x1
        self.dw2 = self.da1 * self.x2
        self.db1 = self.da1

        self.dw3 = self.da2 * self.x1
        self.dw4 = self.da2 * self.x2
        self.db2 = self.da2

    def fit(self, XX, YY, epochs=1, learning_rate=1, display_loss=False, display_weight=False):
        if display_loss:
            loss = {}

        for i in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):
            (
                dw1,dw2,dw3,dw4,dw5,dw6,dw7,dw8,dw9,dw10,dw11,dw12,
                db1,db2,db3,db4,db5,db6,
            ) = [0] * 18
            for X, Y in zip(XX, YY):
                self.grad(X, Y)
                dw1 += self.dw1
                dw2 += self.dw2
                dw3 += self.dw3
                dw4 += self.dw4
                dw5 += self.dw5
                dw6 += self.dw6
                dw7 += self.dw7
                dw8 += self.dw8
                dw9 += self.dw9
                dw10 += self.dw10
                dw11 += self.dw11
                dw12 += self.dw12
                db1 += self.db1
                db2 += self.db2
                db3 += self.db3
                db4 += self.db4
                db2 += self.db5
                db3 += self.db6

            m = XX.shape[0]
            self.w1 -= learning_rate * (dw1 / m)
            self.w2 -= learning_rate * (dw2 / m)
            self.w3 -= learning_rate * (dw3 / m)
            self.w4 -= learning_rate * (dw4 / m)
            self.w5 -= learning_rate * (dw5 / m)
            self.w6 -= learning_rate * (dw6 / m)
            self.w7 -= learning_rate * (dw7 / m)
            self.w8 -= learning_rate * (dw8 / m)
            self.w9 -= learning_rate * (dw9 / m)
            self.w10 -= learning_rate * (dw10 / m)
            self.w11 -= learning_rate * (dw11 / m)
            self.w12 -= learning_rate * (dw12 / m)
            self.b1 -= learning_rate * (db1 / m)
            self.b2 -= learning_rate * (db2 / m)
            self.b3 -= learning_rate * (db3 / m)
            self.b4 -= learning_rate * (db4 / m)
            self.b5 -= learning_rate * (db5 / m)
            self.b6 -= learning_rate * (db6 / m)

            if display_loss:
                YY_pred = self.predict(XX)
                loss[i] = log_loss(np.argmax(YY, axis=1), YY_pred)

        if display_loss:
            # Wtmat1 = [[self.w1, self.w3], [self.w2, self.w4]]
            # Wtmat2 = [[self.w5, self.w6, self.w7, self.w8], [self.w9, self.w10, self.w11, self.w12]]
            
            plt.plot(loss.values())
            plt.xlabel("Epochs")
            plt.ylabel("Log Loss")
            plt.show()

    def predict(self, XX):
        YY_pred = []
        for X in XX:
            Y_pred = self.forward_pass(X)
            YY_pred.append(Y_pred)
        return np.array(YY_pred)
