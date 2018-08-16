# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_multi_percep.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/08/06 16:36:59 by msukhare          #+#    #+#              #
#    Updated: 2018/08/16 16:44:56 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def tanh(z):
    return (((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))))

def forward_prop(X, w1, w2, b1, b2):
    l1 = tanh(w1.dot(X.transpose()) + b1)
    return (sigmoid(w2.dot(l1) + b2))

def back_prop(X, Y, w1, w2, b1, b2, m):
    l1 = tanh(w1.dot(X.transpose()) + b1)
    l2 = sigmoid(w2.dot(l1) + b2)
    dl2 = l2 - Y.transpose()
    dw2 = ((1 / m) * dl2.dot(l1.transpose()))
    db2 = ((1 / m) * np.sum(dl2, axis=1, keepdims=True))
    dl1 = ((w2.transpose().dot(dl2)) * (1 - (tanh(w1.dot(X.transpose()) + b1)**2)))
    dw1 = ((1 / m) * dl1.dot(X))
    db1 = ((1 / m) * np.sum(dl1, axis=1, keepdims=True))
    return (dw2, dw1, db2, db1)

def cost_fct(X, Y, w1, w2, b1, b2, m):
    sum = 0
    pred = forward_prop(X, w1, w2, b1, b2)
    for i in range(int(m)):
        if (Y[i] == 1):
            sum += np.log(pred[0][i])
        else:
            sum += np.log(1 - pred[0][i])
    return ((-(1 / m) * sum))

def main():
    check_argv()
    data, Y = read_file()
    X = scaling_feat(data)
    m = X.shape[0]
    epsilon = 0.0001
#   thetas1 = (np.random.rand(36, X.shape[1]) * (2 * epsilon) - epsilon)
    w1 = (np.random.rand(100, X.shape[1]) * epsilon)
    w2 = (np.random.rand(1, 100) * epsilon)
    b1 = np.zeros((100, 1), dtype=float)
    b2 = np.zeros((1, 1), dtype=float)
    index = []
    res_cost = []
    alpha = 0.009
    for i in range(10000):
        dw2, dw1, db2, db1 = back_prop(X, Y, w1, w2, b1, b2, m)
        w1 = w1 - (alpha * dw1)
        w2 = w2 - (alpha * dw2)
        b1 = b1 - (alpha * db1)
        b2 = b2 - (alpha * db2)
        index.append(i)
        res_cost.append(cost_fct(X, Y, w1, w2, b1, b2, m))
    plt.plot(index, res_cost, color='red')
    plt.show()
    print(forward_prop(X, w1, w2, b1, b2))
    print(Y.transpose())

if (__name__ == "__main__"):
    main()
