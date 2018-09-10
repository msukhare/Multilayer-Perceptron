# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neural_network.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/07 09:18:00 by msukhare          #+#    #+#              #
#    Updated: 2018/09/10 16:43:54 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from activation_function import activate_layer
from activation_function import deri_func
import matplotlib.pyplot as plt

###initialisation des thetas
def create_w(nb_layers, nb_neurones, ep, nb_features):
    w = []
    w.append((np.random.rand(nb_neurones[0], nb_features) * ep))
    for i in range(1, int(nb_layers)):
        w.append((np.random.rand(nb_neurones[i], nb_neurones[i - 1]) * ep))
    return (w)

def create_bias(nb_layers, nb_neurones):
    bias = []
    for i in range(int(nb_layers)):
        bias.append(np.zeros((nb_neurones[i], 1), dtype=float))
    return (bias)
##Fin initialisation

class neural_network:

    def __init__(self):
        self.alpha = 0.02
        self.epoch = 1000
        self.epsilon_rand = 0.01
        self.nb_layer = 3
        self.nb_neurones = [36, 15, 2]
        self.activate_func = [12, 12, 2] ##1 = sigmoid, 2 = softmax, 10 = tanh, 11 = relu, 12 = leaky_relu
        self.nb_features = 31
        self.w = create_w(self.nb_layer, self.nb_neurones, self.epsilon_rand, self.nb_features)
        self.bias = create_bias(self.nb_layer, self.nb_neurones)

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
        dw = self.get_derivate_w(dl, ls, m)
        dbias = self.get_derivate_bias(dl, m)
        self.gradient_descent(dw, dbias)

    ##cost function for softmax, don't working for sigmoid
    def cost_function(self, Y, predicted, m):
        sum = 0
        print(np.shape(Y))
        print("predicted = ", np.shape(predicted))
        for i in range(int(m)):
            if (Y[0][i] == 1):
                sum += -np.log(predicted[self.nb_layer][0][i])
            else:
                sum += -np.log(predicted[self.nb_layer][1][i])
        return (((1 / m) * sum))

    ##functoion to train thetas
    def train_thetas(self, X_cost, Y_cost, X_train, Y_train):
        m = X_train.shape[0]
        m_cost = X_cost.shape[0]
        cost_funct = []
        index = []
        for i in range(self.epoch):
            self.back_propagation(X_train, Y_train, m)
            cost_funct.append(self.cost_function(Y_cost.transpose(), self.forward_prop(X_cost), m_cost))
            index.append(i)
        plt.plot(index, cost_funct, color='red')
        plt.show()
    
