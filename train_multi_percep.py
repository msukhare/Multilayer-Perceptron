# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_multi_percep.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/08/06 16:36:59 by msukhare          #+#    #+#              #
#    Updated: 2018/08/10 16:52:50 by msukhare         ###   ########.fr        #
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

def forward_prop(X, thetas1, thetas2, thetas3, bias1, bias2, bias3):
    l1 = sigmoid(thetas1.dot(X.transpose()) + bias1)
    l2 = sigmoid(thetas2.dot(l1) + bias2)
    return (sigmoid(thetas3.dot(l2) + bias3))

def sum_mat(dlayer):
    row = dlayer.shape[0]
    col = dlayer.shape[1]
    ret = np.zeros((row, 1), dtype=float)
    for i in range(int(row)):
        for j in range(int(col)):
            ret[i][0] += dlayer[i][j]
    return (ret)
#dbias3 = ((1 / m) * sum_mat(dlayer3))

def back_prop(X, Y, thetas1, thetas2, thetas3, bias1, bias2, bias3, m):
    l1 = sigmoid(thetas1.dot(X.transpose()) + bias1)
    l2 = sigmoid(thetas2.dot(l1) + bias2)
    l3 = sigmoid(thetas3.dot(l2) + bias3)
    dlayer3 = l3 - Y.transpose()
    dlayer2 = ((thetas3.transpose().dot(dlayer3)) * (l2 * (1 - l2)))
    dlayer1 = ((thetas2.transpose().dot(dlayer2)) * (l1 * (1 - l1)))
    dtethas3 = ((1 / m) * (dlayer3.dot(l2.transpose())))
    dtethas2 = ((1 / m) * (dlayer2.dot(l1.transpose())))
    dtethas1 = ((1 / m) * (dlayer1.dot(X)))
    dbias3 = np.sum(dlayer3, axis=1, keepdims=True)
    dbias2 = np.sum(dlayer2, axis=1, keepdims=True)
    dbias1 = np.sum(dlayer2, axis=1, keepdims=True)
    thetas1 -= (0.0000000000001 * dtethas1)
    thetas2 -= (0.0000000000001 * dtethas2)
    thetas3 -= (0.0000000000001 * dtethas3)
    bias1 -= (0.0000000000001 * dbias1)
    bias2 -= (0.0000000000001 * dbias2)
    bias3 -= (0.0000000000001 * dbias3)

def cost_fct(X, Y, thetas1, thetas2, thetas3, bias1, bias2, bias3, m):
    sum = 0
    predict = forward_prop(X, thetas1, thetas2, thetas3, bias1, bias2, bias3)
    for i in range(int(m)):
        if (Y[i] == 1):
            sum += np.log(predict[0][i])
        else:
            sum += np.log(1 - predict[0][i])
    return (-(1 / m) * sum)

def main():
    check_argv()
    data, Y = read_file()
    X = scaling_feat(data)
    m = X.shape[0]
    epsilon = 0.01
    thetas1 = (np.random.rand(36, X.shape[1]) * (2 * epsilon) - epsilon)
    thetas2 = (np.random.rand(36, 36) * (2 * epsilon) - epsilon)
    thetas3 = (np.random.rand(1, 36) * (2 * epsilon) - epsilon)
    bias1 = (np.random.rand(36, 1) * (2 * epsilon) - epsilon)
    bias2 = (np.random.rand(36, 1) * (2 * epsilon) - epsilon)
    bias3 = (np.random.rand(1, 1) * (2 * epsilon) - epsilon)
    res_cost = []
    index = []
    for i in range(2000):
        back_prop(X, Y, thetas1, thetas2, thetas3, bias1, bias2, bias3, m)
        index.append(i)
        res_cost.append(cost_fct(X, Y, thetas1, thetas2, thetas3, bias1, bias2, bias3, m))
    plt.plot(index, res_cost, color='red')
    plt.show()
    print(forward_prop(X, thetas1, thetas2, thetas3, bias1, bias2, bias3))


if (__name__ == "__main__"):
    main()
