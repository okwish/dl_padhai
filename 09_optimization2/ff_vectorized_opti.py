# FF network with multiple(vector) input, multiple(vector) ouput
# multiple layers
# vectorized implementation
# opti algorithms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, log_loss

from tqdm import notebook


class FFVectorOpti:
    def __init__(self, Wmat1, Wmat2):
        self.params = {} # all params dictionary
        self.params["Wmat1"] = Wmat1.copy()
        self.params["Wmat2"] = Wmat2.copy()
        self.params["Bvec1"] = np.zeros((1, 2))
        self.params["Bvec2"] = np.zeros((1, 4))

        self.num_layers = 2 # not counting input
        # more like "weight-layers"

        self.gradients = {} # all gradients dictionary

        # for momentum, etc.
        self.update_params = {}
        self.prev_update_params = {}
        for i in range(1, self.num_layers + 1):
            self.update_params["v_w" + str(i)] = 0
            self.update_params["v_b" + str(i)] = 0
            self.update_params["m_b" + str(i)] = 0
            self.update_params["m_w" + str(i)] = 0
            self.prev_update_params["v_w" + str(i)] = 0
            self.prev_update_params["v_b" + str(i)] = 0

    def forward_activation(self, AA):
        return 1.0 / (1.0 + np.exp(-AA))

    def grad_activation(self, AA):
        return AA * (1 - AA)

    def softmax(self, AA):
        EE = np.exp(AA)
        return EE / np.sum(EE, axis=1).reshape(-1, 1)

    # params passsed
    # giving option to find at other weights, etc also.

    def forward_pass(self, XX, params=None):
        if params is None:
            params = self.params

        self.AA1 = np.matmul(XX, params["Wmat1"]) + params["Bvec1"]  # (N, 2) * (2, 2) -> (N, 2)
        self.HH1 = self.forward_activation(self.AA1)  # (N, 2)
        self.AA2 = np.matmul(self.HH1, params["Wmat2"]) + params["Bvec2"]  # (N, 2) * (2, 4) -> (N, 4)
        self.HH2 = self.softmax(self.AA2)  # (N, 4)
        return self.HH2

    # total gradient(all datapoints) - found at once
    def grad(self, XX, YY, params=None):
        if params is None:
            params = self.params

        self.forward_pass(XX, params)
        m = XX.shape[0]
        self.gradients["dAA2"] = self.HH2 - YY  
        # (N, 4) - (N, 4) -> (N, 4)
        self.gradients["dWmat2"] = np.matmul(self.HH1.T, self.gradients["dAA2"])  
        # (2, N) * (N, 4) -> (2, 4)
        self.gradients["dBvec2"] = np.sum(self.gradients["dAA2"], axis=0).reshape(1, -1)  
        # (N, 4) -> (1, 4)
        self.gradients["dHH1"] = np.matmul(self.gradients["dAA2"], params["Wmat2"].T) 
        # (N, 4) * (4, 2) -> (N, 2)
        self.gradients["dAA1"] = np.multiply( self.gradients["dHH1"], self.grad_activation(self.HH1) )  
        # (N, 2) .* (N, 2) -> (N, 2)
        self.gradients["dWmat1"] = np.matmul( XX.T, self.gradients["dAA1"] )  
        # (2, N) * (N, 2) -> (2, 2)
        self.gradients["dBvec1"] = np.sum(self.gradients["dAA1"], axis=0).reshape(1, -1)  
        # (N, 2) -> (1, 2)

    def fit(self,
        XX,YY,
        epochs=1,algo="GD",
        display_loss=False,
        eta=1,mini_batch_size=100,
        eps=1e-8,beta=0.9,
        beta1=0.9,beta2=0.9,gamma=0.9,
    ):
        if display_loss:
            loss = {}


        for num_epoch in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):

            m = XX.shape[0] #total number of datapoints

            if algo == "GD":
                self.grad(XX, YY)
                # update parameters:
                for i in range(1, self.num_layers + 1):
                    self.params["Wmat" + str(i)] -= eta * (self.gradients["dWmat" + str(i)] / m)
                    self.params["Bvec" + str(i)] -= eta * (self.gradients["dBvec" + str(i)] / m)

            elif algo == "MiniBatch":
                for k in range(0, m, mini_batch_size):

                    self.grad(XX[k : k + mini_batch_size], YY[k : k + mini_batch_size])
                    # CALLING GRAD ON A MINI TENSOR.(mini batch)
                    # finding gradient over those datapoints only

                    # update parameters:
                    for i in range(1, self.num_layers + 1):
                        self.params["Wmat" + str(i)] -= eta * (self.gradients["dWmat" + str(i)] / mini_batch_size)
                        self.params["Bvec" + str(i)] -= eta * (self.gradients["dBvec" + str(i)] / mini_batch_size)

            elif algo == "Momentum":
                self.grad(XX, YY)
                for i in range(1, self.num_layers + 1):
                    # ELEMENT WISE
                    # operation like +,- broadcast elementwise
                    self.update_params["v_w" + str(i)] = (
                        gamma * self.update_params["v_w" + str(i)] + eta * (self.gradients["dWmat" + str(i)] / m)
                    )
                    self.update_params["v_b" + str(i)] = (
                        gamma * self.update_params["v_b" + str(i)] + eta * (self.gradients["dBvec" + str(i)] / m)
                    )

                    # update parameters:
                    self.params["Wmat" + str(i)] -= self.update_params["v_w" + str(i)]
                    self.params["Bvec" + str(i)] -= self.update_params["v_b" + str(i)]

            elif algo == "NAG":
                temp_params = {}
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] = (
                        gamma * self.prev_update_params["v_w" + str(i)]
                    )
                    self.update_params["v_b" + str(i)] = (
                        gamma * self.prev_update_params["v_b" + str(i)]
                    )
                    temp_params["Wmat" + str(i)] = (
                        self.params["Wmat" + str(i)] - self.update_params["v_w" + str(i)]
                    )
                    temp_params["Bvec" + str(i)] = (
                        self.params["Bvec" + str(i)] - self.update_params["v_b" + str(i)]
                    )

                # gradient at temp parameters
                self.grad(XX, YY, temp_params)
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] = ( 
                        gamma * self.update_params["v_w" + str(i)] + eta * (self.gradients["dWmat" + str(i)] / m)
                    )
                    self.update_params["v_b" + str(i)] = ( 
                        gamma * self.update_params["v_b" + str(i)] + eta * (self.gradients["dBvec" + str(i)] / m)
                    )

                    # update parameters:
                    self.params["Wmat" + str(i)] -= eta * (self.update_params["v_w" + str(i)])
                    self.params["Bvec" + str(i)] -= eta * (self.update_params["v_b" + str(i)])
                self.prev_update_params = self.update_params

            elif algo == "AdaGrad":
                self.grad(XX, YY)
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] += (
                        self.gradients["dWmat" + str(i)] / m
                    ) ** 2
                    self.update_params["v_b" + str(i)] += (
                        self.gradients["dBvec" + str(i)] / m
                    ) ** 2

                    # update parameters:
                    self.params["Wmat" + str(i)] -= ( 
                        (eta / (np.sqrt(self.update_params["v_w" + str(i)]) + eps)) * (self.gradients["dWmat" + str(i)] / m)
                    )
                    self.params["Bvec" + str(i)] -= ( 
                        (eta / (np.sqrt(self.update_params["v_b" + str(i)]) + eps)) * (self.gradients["dBvec" + str(i)] / m)
                    )

            elif algo == "RMSProp":
                self.grad(XX, YY)
                for i in range(1, self.num_layers + 1):
                    self.update_params["v_w" + str(i)] = (
                        beta * self.update_params["v_w" + str(i)] + (1 - beta) * ((self.gradients["dWmat" + str(i)] / m) ** 2)
                    )
                    self.update_params["v_b" + str(i)] = ( 
                        beta * self.update_params["v_b" + str(i)] + (1 - beta) * ((self.gradients["dBvec" + str(i)] / m) ** 2)
                    )

                    # update parameters:
                    self.params["Wmat" + str(i)] -= ( 
                        (eta / (np.sqrt(self.update_params["v_w" + str(i)] + eps))) * (self.gradients["dWmat" + str(i)] / m)
                    )
                    self.params["Bvec" + str(i)] -= ( 
                        (eta / (np.sqrt(self.update_params["v_b" + str(i)] + eps))) * (self.gradients["dBvec" + str(i)] / m)
                    )

            elif algo == "Adam":
                self.grad(XX, YY)
                num_updates = 0
                for i in range(1, self.num_layers + 1):
                    num_updates += 1
                    self.update_params["m_w" + str(i)] = (
                        beta1 * self.update_params["m_w" + str(i)] + (1 - beta1) * (self.gradients["dWmat" + str(i)] / m)
                    )
                    self.update_params["v_w" + str(i)] = (
                        beta2 * self.update_params["v_w" + str(i)] + (1 - beta2) * ((self.gradients["dWmat" + str(i)] / m) ** 2)
                    )

                    m_w_hat = self.update_params["m_w" + str(i)] / (1 - np.power(beta1, num_updates))
                    v_w_hat = self.update_params["v_w" + str(i)] / (1 - np.power(beta2, num_updates))

                    self.params["Wmat" + str(i)] -= (eta / np.sqrt(v_w_hat + eps)) * m_w_hat

                    self.update_params["m_b" + str(i)] = (
                        beta1 * self.update_params["m_b" + str(i)] + (1 - beta1) * (self.gradients["dBvec" + str(i)] / m)
                    )
                    self.update_params["v_b" + str(i)] = (
                        beta2 * self.update_params["v_b" + str(i)] + (1 - beta2) * ((self.gradients["dBvec" + str(i)] / m) ** 2)
                    )

                    m_b_hat = self.update_params["m_b" + str(i)] / (1 - np.power(beta1, num_updates))
                    v_b_hat = self.update_params["v_b" + str(i)] / (1 - np.power(beta2, num_updates))

                    self.params["Bvec" + str(i)] -= (eta / np.sqrt(v_b_hat + eps)) * m_b_hat
           
            if display_loss:
                Y_pred = self.predict(XX)
                loss[num_epoch] = log_loss(np.argmax(YY, axis=1), Y_pred)

        if display_loss:       
            plt.figure(figsize=(3,2)) # for a smaller plot     
            plt.plot(loss.values(), "-o", markersize=5)
            plt.xlabel("Epochs")
            plt.ylabel("Log Loss")
            plt.show()

    def predict(self, XX):
        YY_pred = self.forward_pass(XX)
        return np.array(YY_pred).squeeze()
