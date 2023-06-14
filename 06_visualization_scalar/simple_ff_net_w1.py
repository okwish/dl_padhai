# FF net class with one hidden layer (3 neurons: 2 in hiddlen layer and 1 in ouput)
# sigmoid neurons

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from tqdm import notebook

class SimpleFFNetW1:

    # PARAMETERS

    def __init__(self):
        self.w1 = np.random.randn()
        # only w1 will be optimized, rest remain same
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    # FORWARD PASS

    def forward_pass(self, X):
        self.x1, self.x2 = X #input also set as class variables
        # computing things one by one.
        self.a1 = self.w1 * self.x1 + self.w2 * self.x2 + self.b1
        self.h1 = self.sigmoid(self.a1)
        self.a2 = self.w3 * self.x1 + self.w4 * self.x2 + self.b2
        self.h2 = self.sigmoid(self.a2)
        self.a3 = self.w5 * self.h1 + self.w6 * self.h2 + self.b3
        self.h3 = self.sigmoid(self.a3)
        return self.h3

    # GRADIENT

    def grad(self, X, y):
        self.forward_pass(X) #updates self - ai, hi

        # self.dw5 = (self.h3 - y) * self.h3 * (1 - self.h3) * self.h1
        # self.dw6 = (self.h3 - y) * self.h3 * (1 - self.h3) * self.h2
        # self.db3 = (self.h3 - y) * self.h3 * (1 - self.h3)

        # only updating one weight
        self.dw1 = (
            (self.h3 - y) * self.h3 * (1 - self.h3) * self.w5 * self.h1 * (1 - self.h1) * self.x1
        )
        # self.dw2 = (
        #     (self.h3 - y) * self.h3 * (1 - self.h3) * self.w5 * self.h1 * (1 - self.h1) * self.x2
        # )
        # self.db1 = (
        #     (self.h3 - y) * self.h3 * (1 - self.h3) * self.w5 * self.h1 * (1 - self.h1)
        # )
        # self.dw3 = (
        #     (self.h3 - y) * self.h3 * (1 - self.h3) * self.w6 * self.h2 * (1 - self.h2) * self.x1
        # )
        # self.dw4 = (
        #     (self.h3 - y) * self.h3 * (1 - self.h3) * self.w6 * self.h2 * (1 - self.h2) * self.x2
        # )
        # self.db2 = (
        #     (self.h3 - y) * self.h3 * (1 - self.h3) * self.w6 * self.h2 * (1 - self.h2)
        # )


    # FIT

    def fit(self, XX, Y, epochs=1, learning_rate=1, display_loss=False): # removed 'initialise'
        # initialise w, b
        # if initialise:
        #     # if this is true, a new initialisation in each epoch. - otherwise cummulative.
        #     self.w1 = np.random.randn()
        #     self.w2 = np.random.randn()
        #     self.w3 = np.random.randn()
        #     self.w4 = np.random.randn()
        #     self.w5 = np.random.randn()
        #     self.w6 = np.random.randn()
        #     self.b1 = 0
        #     self.b2 = 0
        #     self.b3 = 0

        if display_loss:
            loss = {}
            w1 = {} #dictionary to track w1 values

        for i in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):
            # dw1, dw2, dw3, dw4, dw5, dw6, db1, db2, db3 = [0] * 9
            dw1 = 0
            for X, y in zip(XX, Y):
                self.grad(X, y)
                # callind grad - computes the gradient(with that input) and the values are now in self.dwi
                # update with average of gradient over all input.
                # cummulate them into variable dwi, iterating through each datapoint - as we need average.
                dw1 += self.dw1
                # dw2 += self.dw2
                # dw3 += self.dw3
                # dw4 += self.dw4
                # dw5 += self.dw5
                # dw6 += self.dw6
                # db1 += self.db1
                # db2 += self.db2
                # db3 += self.db3

            # updating parameters after coming out of all the datapoints.
            m = XX.shape[0]
            # dividing by m -> average.
            self.w1 -= learning_rate * dw1 / m
            # self.w2 -= learning_rate * dw2 / m
            # self.w3 -= learning_rate * dw3 / m
            # self.w4 -= learning_rate * dw4 / m
            # self.w5 -= learning_rate * dw5 / m
            # self.w6 -= learning_rate * dw6 / m
            # self.b1 -= learning_rate * db1 / m
            # self.b2 -= learning_rate * db2 / m
            # self.b3 -= learning_rate * db3 / m

            if display_loss:
                w1[i] = self.w1 # store w1 values
                Y_pred = self.predict(XX)
                loss[i] = mean_squared_error(Y_pred, Y)

        if display_loss:
            # plt.plot(loss.values())
            # plt.xlabel("Epochs")
            # plt.ylabel("Mean Squared Error")
            # plt.show()

            # plot both w1 values and loss values
            plt.tight_layout() #for spacing between subplots
      
            plt.subplot(2,1,1)
            plt.plot(w1.values())
            plt.xlabel('Epochs')
            plt.ylabel('W1')
            
            plt.subplot(2,1,2)
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            
            plt.show()

    # PREDICT

    def predict(self, XX):
        Y_pred = []
        for X in XX:
            y_pred = self.forward_pass(X)
            Y_pred.append(y_pred)
        return np.array(Y_pred)