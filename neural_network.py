# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neural_network.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/07 09:18:00 by msukhare          #+#    #+#              #
#    Updated: 2018/09/09 21:48:30 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from activation_function import activate_layer
from activation_function import deri_func

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
        self.activate_func = [10, 10, 2] ##1 = sigmoid, 2 = softmax, 10 = tanh, 11 = relu, 12 = leaky_relu
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

    ###Backpropagation and update thetas and bias with gradient descent
    def back_propagation(self, X, Y):
        ls = self.forward_prop(X)
        dl = []
        m = X.shape[0]
        dl.append((ls[self.nb_layer] - Y.transpose()))
        j = 0
        for i in range(int(self.nb_layer - 1), 0, -1):
            dl.append(((self.w[i].transpose().dot(dl[j])) * \
                deri_func((self.w[i - 1].dot(ls[i - 1]) + self.bias[i - 1]),
                    self.activate_func[i - 1])))
            j += 1

    ##cost function for softmax, don't working for sigmoid
    def cost_function(self, Y, predicted):
        m = Y.shape[1] ##number of exemple 
        sum = 0
        for i in range(int(m)):
            if (Y[0][i] == 1):
                sum += -np.log(predicted[self.nb_layer][0][i])
            else:
                sum += -np.log(predicted[self.nb_layer][1][i])
        return (((1 / m) * sum))

    ##function to train thetas
    def train_thetas(self, X_cost, Y_cost, X_train, Y_train):
        for i in range(self.epoch):
            back_propagation(X_train, Y_train)
            cost_function(Y_cost.transpose(), forward_prop(X_cost))


