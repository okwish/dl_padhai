# FF network with multiple(vector) input, multiple(vector) ouput
# multiple layers
# vectorized implementation
# opti algorithms - GD
# regularization
# initializations
# activations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from numpy.linalg import norm
from tqdm import notebook


class FFNet:
    def __init__(self, 
        num_hidden=2,
        init_method="xavier", 
        activation_function="sigmoid", 
        leaky_slope=0.1):

        self.params = {} # all params dictionary
        self.num_layers = 2 # like "weight-layers"
        self.layer_sizes = [2, num_hidden, 3] #3 classes(output vector)
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope
        np.random.seed(0)

        # INITIALISATIONS

        np.random.seed(0)
        if init_method == "random":
            for i in range(1, self.num_layers + 1):
                self.params["Wmat" + str(i)] = np.random.randn(
                    self.layer_sizes[i - 1], self.layer_sizes[i]
                )
                self.params["Bvec" + str(i)] = np.random.randn(1, self.layer_sizes[i])

        elif init_method == "he":
            for i in range(1, self.num_layers + 1):
                self.params["Wmat" + str(i)] = np.random.randn(
                    self.layer_sizes[i - 1], self.layer_sizes[i]
                ) * np.sqrt(2 / self.layer_sizes[i - 1])
                self.params["Bvec" + str(i)] = np.random.randn(1, self.layer_sizes[i])

        elif init_method == "xavier":
            for i in range(1, self.num_layers + 1):
                self.params["Wmat" + str(i)] = np.random.randn(
                    self.layer_sizes[i - 1], self.layer_sizes[i]
                ) * np.sqrt(1 / self.layer_sizes[i - 1])
                self.params["Bvec" + str(i)] = np.random.randn(1, self.layer_sizes[i])

        # elif init_method == "zeros":
        #     for i in range(1, self.num_layers + 1):
        #         self.params["Wmat" + str(i)] = np.zeros(
        #             (self.layer_sizes[i - 1], self.layer_sizes[i])
        #         )
        #         self.params["Bvec" + str(i)] = np.zeros((1, self.layer_sizes[i]))


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

    # ACTIVATION FUNCTIONS

    def forward_activation(self, AA):
        if self.activation_function == "sigmoid":
            return 1.0 / (1.0 + np.exp(-AA))
        elif self.activation_function == "tanh":
            return np.tanh(AA)
        elif self.activation_function == "relu":
            return np.maximum(0, AA)
        elif self.activation_function == "leaky_relu":
            return np.maximum(self.leaky_slope * AA, AA)

    # CORRESPONDING DERIVATIVE FORMULAS

    def grad_activation(self, AA):
        if self.activation_function == "sigmoid":
            return AA * (1 - AA)
        elif self.activation_function == "tanh":
            return 1 - np.square(AA)
        elif self.activation_function == "relu":
            return 1.0 * (
                AA > 0
            )  # AA>0 . return element wise bool of the condition. multiplying by 1 cast to int.
        elif self.activation_function == "leaky_relu":
            DD = np.zeros_like(AA)
            DD[AA <= 0] = self.leaky_slope
            DD[AA > 0] = 1
            return DD
    

    # to find accuracy with curent-parameters after each epoch
    def get_accuracy(self, XX_train, Y_train, XX_val, Y_val):

        YY_pred_train = self.predict(XX_train)
        Y_pred_train = np.argmax(YY_pred_train, 1)

        YY_pred_val = self.predict(XX_val)
        Y_pred_val = np.argmax(YY_pred_val, 1)

        accuracy_train = accuracy_score(Y_pred_train, Y_train)
        accuracy_val = accuracy_score(Y_pred_val, Y_val)
        return accuracy_train, accuracy_val

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
        XX,YY, #XX_train, YY_OH_train
        XX_val, Y_val,
        epochs=1,
        l2_norm=False, lambda_val=0.8,
        display_loss=False,
        eta=1,
    ):
        # dictionary of train , validation accuracy
        # i-th key => accuracies at i-th epoch
        train_accuracies = {}
        val_accuracies = {}

        if display_loss:
            loss = []
            weight_mag = [] # list of norm of weights after each epoch
            

        for num_epoch in notebook.tqdm(range(epochs), total=epochs, unit="epoch"):

            m = XX.shape[0] #total number of datapoints


            self.grad(XX, YY)
            for i in range(1, self.num_layers + 1):
                if l2_norm:
                    # extra term - which is derivative of L2 norm term in loss fn.
                    # lambda_val -> coeff
                    self.params["Wmat" + str(i)] -= (
                        (eta * lambda_val) / m * self.params["Wmat" + str(i)] + eta * (self.gradients["dWmat" + str(i)] / m)
                    )
                else:
                    self.params["Wmat" + str(i)] -= eta * (self.gradients["dWmat" + str(i)] / m)

                self.params["Bvec" + str(i)] -= eta * (self.gradients["dBvec" + str(i)] / m)



            # find both train and val accuracy after each epoch

            XX_train = XX
            YY_OH_train = YY
            #one hot to labels
            Y_train = np.argmax(YY_OH_train, axis=1)
            # instead of passing Y_train also from outside

            train_accuracy, val_accuracy = self.get_accuracy(XX_train, Y_train, XX_val, Y_val)
            train_accuracies[num_epoch] = train_accuracy
            val_accuracies[num_epoch] = val_accuracy

            if display_loss:
                YY_pred = self.predict(XX)
                loss.append(log_loss(np.argmax(YY, axis=1), YY_pred))
                # parameter-norm after each epoch
                weight_mag.append(
                    (
                        norm(self.params["Wmat1"]) + norm(self.params["Wmat2"])
                        + norm(self.params["Bvec1"]) + norm(self.params["Bvec2"])
                    )/ 18
                )

        # plot accuracies (train and val) after each epoch
        plt.plot(train_accuracies.values(), label="Train accuracy")
        plt.plot(val_accuracies.values(), label="Validation accuracy")
        plt.plot(np.ones((epochs, 1)) * 0.9)  # plotting a good accuracy to expect.(benchmark)
        plt.plot(np.ones((epochs, 1)) * 0.33)  # baseline - one predicting one class all the time.
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        # plot loss with epochs - and parameters-norm with epochs 
        # in same plot using twinx

        # plot two things(different y-range) in same plot.(secondary y axis) - using twinx
        # loss, weight mag.
        if display_loss:
            # subplot returns two handlers - one for figure and other for axis
            # (can do stuff with the handler)
            fig, ax1 = plt.subplots()
            color = "tab:red"
            # setting labels using the axis handler
            ax1.set_xlabel("epochs")
            ax1.set_ylabel("Log Loss", color=color)
            # plot loss.
            ax1.plot(loss, "-o", color=color)
            ax1.tick_params(axis="y", labelcolor=color)

            # ax2 is a new handler created as a twin of ax1.
            ax2 = ax1.twinx()
            color = "tab:blue"
            # then can work with ax2 equivalently.
            ax2.set_ylabel("Weight Magnitude", color=color) # we already handled the x-label with ax1
            ax2.plot(weight_mag, "-*", color=color)
            ax2.tick_params(axis="y", labelcolor=color)
            fig.tight_layout()  # to ensure there is not clipping.
            plt.show()


    def predict(self, XX):
        YY_pred = self.forward_pass(XX)
        return np.array(YY_pred).squeeze()
