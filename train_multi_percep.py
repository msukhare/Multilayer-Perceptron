# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_multi_percep.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/08/06 16:36:59 by msukhare          #+#    #+#              #
#    Updated: 2018/08/07 10:57:12 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import numpy as np
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

def forward_prop(X, thetas1, thetas2, thetas3):
    l2 = sigmoid(np.dot(thetas1, X))
    l3 = sigmoid(np.dot(thetas2, l2))
    return (sigmoid(np.dot(thetas3, l3)))

def get_summ(X, Y, thetas1, thetas2, thetas3, m):
    summ = 0
    for i in range(int(m)):
        if (Y[i] == 1):
            summ += np.log(forward_prop(X[i], thetas1, thetas2, thetas3))
        else:
            summ += np.log(1 - forward_prop(X[i], thetas1, thetas2, thetas3))
    return (summ)

def cost_fct(X, Y, thetas1, thetas2, thetas3):
    m = X.shape[0]
    return (-(1 / m) * get_summ(X, Y, thetas1, thetas2, thetas3, m))

def back_prop(X, Y, thetas1, thetas2, thetas3):
    m = X.shape[0]
    capital_greek1 = np.zeros((36, 31), dtype=float)
    capital_greek2 = np.zeros((36, 36), dtype=float)
    capital_greek3 = np.zeros((1, 36), dtype=float)
    for i in range(int(m)):
        l1 = X[i]
        l2 = sigmoid(np.dot(thetas1, l1))
        l3 = sigmoid(np.dot(thetas2, l2))
        l4 = sigmoid(np.dot(thetas3, l3))
        lower_case_delta4 = l4 - Y[i]
        lower_case_delta3 = (thetas3.transpose().dot(lower_case_delta4)).dot(l3) * (1 - l3)
        lower_case_delta2 = (thetas2.transpose().dot(lower_case_delta3)).dot(l2) * (1 - l2)
        lower_case_delta2 = np.reshape(lower_case_delta2, (36, 1))
        lower_case_delta3 = np.reshape(lower_case_delta3, (36, 1))
        lower_case_delta4 = np.reshape(lower_case_delta4, (1, 1))
        l1 = np.reshape(l1, (1, 31))
        capital_greek1 += lower_case_delta2.dot(l1)
        l2 = np.reshape(l2, (1, 36))
        capital_greek2 += lower_case_delta3.dot(l2)
        l3 = np.reshape(l3, (1, 36))
        capital_greek3 += lower_case_delta4.dot(l3)
    capital_greek1 = 1 / m * capital_greek1
    capital_greek2 = 1 / m * capital_greek2
    capital_greek3 = 1 / m * capital_greek3
    return (capital_greek1, capital_greek2, capital_greek3)

def main():
    check_argv()
    data, Y = read_file()
    X = scaling_feat(data)
    epsilon = 0.01
    thetas1 = (np.random.rand(36, X.shape[1]) * (2 * epsilon) - epsilon)
    thetas2 = (np.random.rand(36, 36) * (2 * epsilon) - epsilon)
    thetas3 = (np.random.rand(1, 36) * (2 * epsilon) - epsilon)
    cost_fct(X, Y, thetas1, thetas2, thetas3)
    cap1, cap2, cap3 = back_prop(X, Y, thetas1, thetas2, thetas3)

if (__name__ == "__main__"):
    main()
