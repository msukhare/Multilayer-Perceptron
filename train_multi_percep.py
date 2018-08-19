# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_multi_percep.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/08/06 16:36:59 by msukhare          #+#    #+#              #
#    Updated: 2018/08/19 16:50:34 by kemar            ###   ########.fr        #
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

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def tanh(z):
    return (((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))))

def forward_prop(X, w1, w2, w3, b1, b2, b3):
    l1 = tanh(w1.dot(X.transpose()) + b1)
    l2 = tanh(w2.dot(l1) + b2)
    return (sigmoid(w3.dot(l2) + b3))

def back_prop(X, Y, w1, w2, w3, b1, b2, b3, m):
    #l1 = tanh(w1.dot(X.transpose()) + b1)
    #l2 = sigmoid(w2.dot(l1) + b2)
    #dl2 = l2 - Y.transpose()
    #dw2 = ((1 / m) * dl2.dot(l1.transpose()))
    #db2 = ((1 / m) * np.sum(dl2, axis=1, keepdims=True))
    #dl1 = ((w2.transpose().dot(dl2)) * (1 - (tanh(w1.dot(X.transpose()) + b1)**2)))
    #dw1 = ((1 / m) * dl1.dot(X))
    #db1 = ((1 / m) * np.sum(dl1, axis=1, keepdims=True))
    l1 = tanh(w1.dot(X.transpose()) + b1)
    l2 = tanh(w2.dot(l1) + b2)
    l3 = sigmoid((w3.dot(l2) + b3))
    dl3 = l3 - Y.transpose()
    dw3 = ((1 / m) * dl3.dot(l2.transpose()))
    db3 = ((1 / m) * np.sum(dl3, axis=1, keepdims=True))
    dl2 = ((w3.transpose().dot(dl3)) * (1 - (tanh(w2.dot(l1)) + b2)**2))
    dw2 = ((1 / m) * dl2.dot(l1.transpose()))
    db2 = ((1 / m) * np.sum(dl2, axis=1, keepdims=True))
    dl1 = ((w2.transpose().dot(dl2)) * (1 - (tanh(w1.dot(X.transpose()) + b1)**2)))
    dw1 = ((1 / m) * dl1.dot(X))
    db1 = ((1 / m) * np.sum(dl1, axis=1, keepdims=True))
    return (dw3, dw2, dw1, db3, db2, db1)

def cost_fct(X, Y, w1, w2, w3, b1, b2, b3, m):
    sum = 0
    pred = forward_prop(X, w1, w2, w3, b1, b2, b3)
    for i in range(int(m)):
        if (Y[i] == 1):
            sum += np.log(pred[0][i])
        else:
            sum += np.log(1 - pred[0][i])
    return ((-(1 / m) * sum))

def gradient_check(X, Y, w1, w2, b1, b2, alpha, m):
    dw2, dw1, db2, db1 = back_prop(X, Y, w1, w2, b1, b2, m)
    dw2 = np.ravel(dw2)
    dw1 = np.ravel(dw1)
    db2 = np.ravel(db2)
    db1 = np.ravel(db1)
    bigdwb = np.reshape(np.concatenate((dw1, db1, dw2, db2)), (1189, 1))
    dapprox = np.zeros((1189, 1), dtype=float)
    k = 0
    ep = 0.0000001
    for i in range(36):
        for j in range(31):
            tmp = w1[i][j]
            w1[i][j] += ep
            plus = cost_fct(X, Y, w1, w2, b1, b2, m)
            w1[i][j] = tmp
            tmp = w1[i][j]
            w1[i][j] -= ep
            neg = cost_fct(X, Y, w1, w2, b1, b2, m)
            w1[i][j] = tmp
            dapprox[k][0] = (plus - neg) / (2 * ep)
            k += 1
    for i in range(36):
        for j in range(1):
            tmp = b1[i][j]
            b1[i][j] += ep
            plus = cost_fct(X, Y, w1, w2, b1, b2, m)
            b1[i][j] = tmp
            tmp = b1[i][j]
            b1[i][j] -= ep
            neg = cost_fct(X, Y, w1, w2, b1, b2, m)
            b1[i][j] = tmp
            dapprox[k][0] = (plus - neg) / (2 * ep)
            k += 1
    for i in range(1):
        for j in range(36):
            tmp = w2[i][j]
            w2[i][j] += ep
            plus = cost_fct(X, Y, w1, w2, b1, b2, m)
            w2[i][j] = tmp
            tmp = w2[i][j]
            w2[i][j] -= ep
            neg = cost_fct(X, Y, w1, w2, b1, b2, m)
            w2[i][j] = tmp
            dapprox[k][0] = (plus - neg) / (2 * ep)
            k += 1
    for i in range(1):
        for j in range(1):
            tmp = b2[i][j]
            b2[i][j] += ep
            plus = cost_fct(X, Y, w1, w2, b1, b2, m)
            b2[i][j] = tmp
            tmp = b2[i][j]
            b2[i][j] -= ep
            neg = cost_fct(X, Y, w1, w2, b1, b2, m)
            b2[i][j] = tmp
            dapprox[k][0] = (plus - neg) / (2 * ep)
            k += 1
    sum1 = 0
    for i in range(int(dapprox.shape[0])):
        sum1 += (dapprox[i][0] - bigdwb[i][0])**2
    sum1 = np.sqrt(sum1)
    sum2 = 0
    for i in range(int(dapprox.shape[0])):
        sum2 += (dapprox[i][0])**2
    sum2 = np.sqrt(sum2)
    sum3 = 0
    for i in range(int(bigdwb.shape[0])):
        sum3 += (bigdwb[i][0])**2
    sum3 = np.sqrt(sum3)
    print((sum1 / (sum2 + sum3)))

def kappa_cohen(vn, vp, fp, fn):
    paccord = ((vp + vn) / (vn + vp + fp + fn))
    pyes = (((vp + fn) / (vn + vp + fp + fn)) * ((vp + fp) / (vn + vp + fp + fn)))
    pno = (((fp + vn) / (vn + vp + fp + fn)) * ((fn + vn) / (vn + vp + fp + fn)))
    phasard = pyes + pno
    print("kappa_cohen\n", ((paccord - phasard) / (1 - phasard)))

def get_quality_classi(pred, Y):
    vn = 0
    vp = 0
    fp = 0
    fn = 0
    for i in range(int(pred.shape[0])):
        if (pred[i][0] >= 0.5 and Y[i][0] == 1):
            vp += 1
        elif (pred[i][0] >= 0.5 and Y[i][0] == 0):
            fp += 1
        elif (pred[i][0] < 0.5 and Y[i][0] == 0):
            vn += 1
        elif (pred[i][0] < 0.5 and Y[i][0] == 1):
            fn += 1
    print("accuracy:\n", ((vp + vn) / int(pred.shape[0])))
    precision = (vp / (vp + fp))
    recall = (vp / (vp + fn))
    print("precision:\n", precision)
    print("recall:\n", recall)
    print("f1:\n", ((2 * (precision * recall)) / (precision + recall)))
    kappa_cohen(vn, vp, fp, fn)
    print("classification error\n", ((fp + fn) / Y.shape[0]))
    print("false alarm rate:\n", (fp / (fp + vn)))
    print("miss rate:\n", (fn / (vp + fn)))

def main():
    check_argv()
    data, Y = read_file()
    X = scaling_feat(data)
    m = X.shape[0]
    epsilon = 0.01
    X_train, X_cost = X[ : floor(m * 0.85)], X[floor(m * 0.85) :]
    Y_train, Y_cost = Y[ : floor(m * 0.85)], Y[floor(m * 0.85) :]
#   thetas1 = (np.random.rand(36, X.shape[1]) * (2 * epsilon) - epsilon)
    w1 = (np.random.rand(36, X.shape[1]) * epsilon)
    w2 = (np.random.rand(36, 36) * epsilon)
    w3 = (np.random.rand(1, 36) * epsilon)
    b1 = np.zeros((36, 1), dtype=float)
    b2 = np.zeros((36, 1), dtype=float)
    b3 = np.zeros((1, 1), dtype=float)
    index = []
    res_cost = []
    alpha = 0.09
    #gradient_check(X_train, Y_train, w1, w2, b1, b2, alpha, floor(0.85 * m))
    for i in range(10000):
        dw3, dw2, dw1, db3, db2, db1 = back_prop(X_train, Y_train, w1, w2, w3, b1, b2, b3, floor(0.85 * m))
        w1 = w1 - (alpha * dw1)
        w2 = w2 - (alpha * dw2)
        w3 = w3 - (alpha * dw3)
        b1 = b1 - (alpha * db1)
        b2 = b2 - (alpha * db2)
        b3 = b3 - (alpha * db3)
        index.append(i)
        res_cost.append(cost_fct(X_cost, Y_cost, w1, w2, w3, b1, b2, b3, floor(0.15 * m)))
    plt.plot(index, res_cost, color='red')
    plt.show()
    pred = forward_prop(X_cost, w1, w2, w3, b1, b2, b3).transpose()
    get_quality_classi(pred, Y_cost)

if (__name__ == "__main__"):
    main()
