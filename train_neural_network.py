# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_neural_network.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/07 11:18:12 by msukhare          #+#    #+#              #
#    Updated: 2018/11/06 15:24:50 by msukhare         ###   ########.fr        #
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
    for key in data:
        if (key != "y"):
            data.fillna(value={key: data[key].mean()}, inplace=True)
    data = data.sample(frac=1, random_state=85).reset_index(drop=True)
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
            X[i][j] = (X[i][j] - desc[key]['mean']) / desc[key]['std']#(desc[key]['max'] - desc[key]['min'])
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

def main():
    check_argv()
    data, Y = read_file()
    X = scaling_feat(data)
    m = X.shape[0]
    X_train, X_cost = X[ : floor(m * 0.80)], X[floor(m * 0.80) :]
    Y_train, Y_cost = Y[ : floor(m * 0.80)], Y[floor(m * 0.80) :]
    Y_train_tmp = copy_and_replace(np.copy(Y_train))
    Y_cost_tmp = copy_and_replace(np.copy(Y_cost))
    Y_train = np.c_[Y_train_tmp, Y_train]
    Y_cost = np.c_[Y_cost_tmp, Y_cost]
    neural_n = neural_network()
    metrics = metrics_for_binary_classification()
    neural_n.initweight([30, 25, 15, 10, 2], [11, 11, 11, 11, 2], data.shape[1], 984, 5)
    #print(neural_n.gradient_checking(X_train, Y_train, 0.0000001))
    neural_n.train_thetas(X_cost, Y_cost, X_train, Y_train)
    layers = neural_n.forward_prop(X_cost)
    pred = layers[neural_n.nb_layer].transpose()
    metrics.confused_matrix_soft_max(pred, Y_cost, 1)
    neural_n.write_architecture()

if __name__ == "__main__":
    main()
