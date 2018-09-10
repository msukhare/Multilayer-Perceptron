# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_neural_network.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/07 11:18:12 by msukhare          #+#    #+#              #
#    Updated: 2018/09/10 16:26:33 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from neural_network import neural_network
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import floor
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
    data = data.sample(frac=1).reset_index(drop=True)
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
    neural_n.train_thetas(X_cost, Y_cost, X_train, Y_train)
    pred = neural_n.forward_prop(X_cost)
    print((pred[neural_n.nb_layer]))

if __name__ == "__main__":
    main()
