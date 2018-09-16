# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neural_network.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/07 09:18:00 by msukhare          #+#    #+#              #
#    Updated: 2018/09/16 13:47:22 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from activation_function import activate_layer
from activation_function import deri_func
import matplotlib.pyplot as plt

##euclidien distance
def euclidien_distance(vec_dwb, dapprox):
    sum1 = np.sqrt(np.sum(((dapprox - vec_dwb)**2), axis=0))
    return ((sum1 / (np.sqrt(np.sum((dapprox**2), axis=0)) + \
            np.sqrt(np.sum((vec_dwb**2), axis=0)))))


class neural_network:

    def __init__(self):
        self.alpha = 0.02
        self.epoch = 40000
        self.epsilon_rand = 0.01
        self.nb_layer = 4
        self.nb_neurones = [36, 25, 15, 2]
        self.activate_func = [12, 12, 12, 2] ##1 = sigmoid, 2 = softmax, 10 = tanh, 11 = relu, 12 = leaky_relu
        self.nb_features = 31
        self.w = []
        self.bias = []

    ##function for make predictions
    def forward_prop(self, X):
        ret = []
        ret.append(X.transpose())
        for i in range(int(self.nb_layer)):
            ret.append(activate_layer((self.w[i].dot(ret[i]) + self.bias[i]),\
                    self.activate_func[i]))
        return (ret)

    ##Update thetas
    def gradient_descent(self, dw, dbias):
        j = (self.nb_layer - 1)
        for i in range(int(self.nb_layer)):
            self.w[j] = (self.w[j] - (self.alpha * dw[i]))
            j -= 1
        j = (self.nb_layer - 1)
        for i in range(int(self.nb_layer)):
            self.bias[j] = (self.bias[j] - (self.alpha * dbias[i]))
            j -= 1

    def get_derivate_layer(self, ls, Y):
        dl = []
        dl.append((ls[self.nb_layer] - Y.transpose()))
        j = 0
        for i in range(int(self.nb_layer - 1), 0, -1):
            dl.append(((self.w[i].transpose().dot(dl[j])) * \
                deri_func((self.w[i - 1].dot(ls[i - 1]) + self.bias[i - 1]),
                    self.activate_func[i - 1])))
            j += 1
        return (dl)

    def get_derivate_w(self, dl, ls, m):
        dw = []
        j = (self.nb_layer - 1)
        for i in range(int(self.nb_layer)):
            dw.append(((1 / m) * (dl[i].dot(ls[j].transpose()))))
            j -= 1
        return (dw)

    def get_derivate_bias(self, dl, m):
        dbias = []
        for i in range(int(self.nb_layer)):
            dbias.append(((1 / m) * (np.sum(dl[i], axis=1, keepdims=True))))
        return (dbias)

    ###Backpropagation and update thetas and bias with gradient descent
    def back_propagation(self, X, Y, m):
        ls = self.forward_prop(X)
        dl = self.get_derivate_layer(ls, Y)
        return (self.get_derivate_w(dl, ls, m), self.get_derivate_bias(dl, m))

    ##cost function for softmax, don't working for sigmoid
    def cost_function(self, Y, predicted, m):
        sum = 0
        for i in range(int(m)):
            if (Y[0][i] == 1):
                sum += -np.log(predicted[self.nb_layer][0][i])
            else:
                sum += -np.log(predicted[self.nb_layer][1][i])
        return (((1 / m) * sum))

    ##function to train thetas
    def train_thetas(self, X_cost, Y_cost, X_train, Y_train):
        m = X_train.shape[0]
        m_cost = X_cost.shape[0]
        validation = []
        index = []
        train = []
        for i in range(self.epoch):
            dw, dbias = self.back_propagation(X_train, Y_train, m)
            self.gradient_descent(dw, dbias)
            validation.append(self.cost_function(Y_cost.transpose(),\
                    self.forward_prop(X_cost), m_cost))
            train.append(self.cost_function(Y_train.transpose(),\
                    self.forward_prop(X_train), m))
            index.append(i)
        plt.plot(index, train, color='red')
        plt.plot(index, validation, color='green')
        plt.show()

###Gradient checking 
## Vectorize derivate of thetas and bias
    def create_vectors(self, dw, dbias):
        ret = np.zeros((0, 1), dtype=float)
        for i in range(int(self.nb_layer - 1), -1, -1):
            dw[i] = np.reshape(np.ravel(dw[i]), ((dw[i].shape[0]\
                    * dw[i].shape[1]), 1))
            dbias[i] = np.reshape(np.ravel(dbias[i]), ((dbias[i].shape[0] \
                    * dbias[i].shape[1]), 1))
            ret = np.reshape(np.concatenate((ret, dw[i], dbias[i])),\
                (((ret.shape[0] * ret.shape[1]) + (dw[i].shape[0]\
                * dw[i].shape[1])) + (dbias[i].shape[0]\
                * dbias[i].shape[1]), 1))
        return (ret, np.zeros((ret.shape[0], ret.shape[1]), dtype=float))

    def compute_with_modif_w(self, epsilon, a, dapprox, X, Y, k, m):
        z = a
        for i in range(int(self.w[k].shape[0])):
            for j in range(int(self.w[k].shape[1])):
                tmp = self.w[k][i][j]
                self.w[k][i][j] += epsilon
                plus = self.cost_function(Y.transpose(), self.forward_prop(X), m)
                self.w[k][i][j] = tmp
                tmp = self.w[k][i][j]
                self.w[k][i][j] -= epsilon
                minus = self.cost_function(Y.transpose(), self.forward_prop(X), m)
                self.w[k][i][j] = tmp
                dapprox[z][0] = ((plus - minus) / (2 * epsilon))
                z += 1
        return ((z - a))

    def compute_with_modif_bias(self, epsilon, a, dapprox, X, Y, k, m):
        z = a
        for i in range(int(self.bias[k].shape[0])):
            for j in range(int(self.bias[k].shape[1])):
                tmp = self.bias[k][i][j]
                self.bias[k][i][j] += epsilon
                plus = self.cost_function(Y.transpose(), self.forward_prop(X), m)
                self.bias[k][i][j] = tmp
                tmp = self.bias[k][i][j]
                self.bias[k][i][j] -= epsilon
                minus = self.cost_function(Y.transpose(), self.forward_prop(X), m)
                self.bias[k][i][j] = tmp
                dapprox[z][0] = ((plus - minus) / (2 * epsilon))
                z += 1
        return ((z - a))

    def gradient_checking(self, X, Y, epsilon):
        dw, dbias = self.back_propagation(X, Y, X.shape[0])
        vec_dwb, dapprox = self.create_vectors(dw, dbias)
        m = X.shape[0]
        a = 0
        for k in range(int(self.nb_layer)):
            a += self.compute_with_modif_w(epsilon, a, dapprox, X, Y, k, m)
            a += self.compute_with_modif_bias(epsilon, a, dapprox, X, Y, k, m)
        return (euclidien_distance(vec_dwb, dapprox))
##End gradient checking
