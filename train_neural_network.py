# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_neural_network.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/07 11:18:12 by msukhare          #+#    #+#              #
#    Updated: 2018/09/16 13:40:52 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from neural_network import neural_network
from metrics_for_binary_classification import metrics_for_binary_classification
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from math import floor
import random
import sys

def check_argv():
    if (len(sys.argv) < 2):
        sys.exit("Need name file with data")
    if (len(sys.argv) > 2):
        sys.exit("too much arguments")

def read_file():
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        sys.exit("Fail to read file, make sure file exist")
    data.dropna(inplace=True)
    data = data.reset_index()
    seed_data = np.random.RandomState(8)
    data = data.sample(frac=1, random_state=seed_data).reset_index(drop=True)
    data.drop(['index'], axis = 1, inplace = True)
    data['y'] = data['y'].map({'B' : 0, 'M' : 1})
    Y = data.iloc[:, 1:2]
    Y = np.array(Y.values, dtype=float)
    data.drop(['y'], axis=1, inplace=True)
    return (data, Y)

def scaling_feat(data):
    X = data.iloc[:, 0:]
    X = np.array(X.values, dtype=float)
    desc = data.describe()
    j = 0
    for key in data:
        for i in range(int(desc[key]['count'])):
            X[i][j] = (X[i][j] - desc[key]['min']) / (desc[key]['max'] - desc[key]['min'])
        j += 1
    return (X)

def copy_and_replace(arr):
    row, col = np.shape(arr)
    for i in range(int(row)):
        for j in range(int(col)):
            if (arr[i][j] == 1):
                arr[i][j] = 0
            else:
                arr[i][j] = 1
    return (arr)

###initialisation des thetas
def create_w(nn):
    w = []
    w.append((np.random.rand(nn.nb_neurones[0], nn.nb_features) * nn.epsilon_rand))
    for i in range(1, int(nn.nb_layer)):
        w.append((np.random.rand(nn.nb_neurones[i], nn.nb_neurones[i - 1]) * nn.epsilon_rand))
    return (w)

def create_bias(nn):
    bias = []
    for i in range(int(nn.nb_layer)):
        bias.append(np.zeros((nn.nb_neurones[i], 1), dtype=float))
    return (bias)
##Fin initialisation

def main():
    check_argv()
    data, Y = read_file()
    X = scaling_feat(data)
    m = X.shape[0]
    epsilon = 0.01
    X_train, X_cost = X[ : floor(m * 0.85)], X[floor(m * 0.85) :]
    Y_train, Y_cost = Y[ : floor(m * 0.85)], Y[floor(m * 0.85) :]
    Y_train_tmp = copy_and_replace(np.copy(Y_train))
    Y_cost_tmp = copy_and_replace(np.copy(Y_cost))
    Y_train = np.c_[Y_train, Y_train_tmp]
    Y_cost = np.c_[Y_cost, Y_cost_tmp]
    neural_n = neural_network()
    metrics = metrics_for_binary_classification()
    if (neural_n.nb_layer <= 0):
        print("nb_layer must be > 0")
    elif (neural_n.nb_layer != len(neural_n.nb_neurones)):
        print("Dimension of nb_neurones must be equals to nb_layer")
    elif (neural_n.nb_layer != len(neural_n.activate_func)):
        print("Dimension of activation_func must be equals to nb_layer")
    else:
        np.random.seed(984)
        neural_n.w = create_w(neural_n)
        neural_n.bias = create_bias(neural_n)
        #print(neural_n.gradient_checking(X_train, Y_train, 0.0000001))
        neural_n.train_thetas(X_cost, Y_cost, X_train, Y_train)
        layers = neural_n.forward_prop(X_cost)
        pred = layers[neural_n.nb_layer].transpose()
        metrics.confused_matrix_soft_max(pred, Y_cost, 1)
        for i in range(int(floor(m * 0.15))):
            print(pred[i][0], pred[i][1], Y_cost[i][0], Y_cost[i][1])

if __name__ == "__main__":
    main()
