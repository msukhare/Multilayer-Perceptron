# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_multi_percep.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/08/06 16:36:59 by msukhare          #+#    #+#              #
#    Updated: 2018/08/14 17:17:33 by msukhare         ###   ########.fr        #
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
    return ((np.exp(z) - np.exp(-z) / (np.exp(z) + np.exp(-z))))

def forward_prop(X, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4):
    l1 = tanh(thetas1.dot(X.transpose()) + bias1)
    l2 = tanh(thetas2.dot(l1) + bias2)
    l3 = tanh(thetas3.dot(l1) + bias3)
    return (sigmoid(thetas4.dot(l3) + bias4))

def back_prop(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m):
    l1 = tanh(thetas1.dot(X.transpose()) + bias1)
    l2 = tanh(thetas2.dot(l1) + bias2)
    l3 = tanh(thetas3.dot(l2) + bias3)
    l4 = sigmoid(thetas4.dot(l3) + bias4)
    dlayer4 = l4 - Y.transpose()
    dlayer3 = ((thetas4.transpose().dot(dlayer4)) * (1 - (tanh(l3))**2))#(l2 * (1 - l2))) Dsigmoid
    dlayer2 = ((thetas3.transpose().dot(dlayer3)) * (1 - (tanh(l2))**2))#(l1 * (1 - l1))) Dsigmoid
    dlayer1 = ((thetas2.transpose().dot(dlayer2)) * (1 - (tanh(l1))**2))#(l1 * (1 - l1))) Dsigmoid
    dtethas4 = ((1 / m) * (dlayer4.dot(l3.transpose())))
    dtethas3 = ((1 / m) * (dlayer3.dot(l2.transpose())))
    dtethas2 = ((1 / m) * (dlayer2.dot(l1.transpose())))
    dtethas1 = ((1 / m) * (dlayer1.dot(X)))
    dbias4 = np.sum(dlayer4, axis=1, keepdims=True)
    dbias3 = np.sum(dlayer3, axis=1, keepdims=True)
    dbias2 = np.sum(dlayer2, axis=1, keepdims=True)
    dbias1 = np.sum(dlayer1, axis=1, keepdims=True)
    thetas1 -= (0.001 * dtethas1)
    thetas2 -= (0.001 * dtethas2)
    thetas3 -= (0.001 * dtethas3)
    thetas4 -= (0.001 * dtethas4)
    bias1 -= (0.001 * dbias1)
    bias2 -= (0.001 * dbias2)
    bias3 -= (0.001 * dbias3)
    bias4 -= (0.001 * dbias4)

def cost_fct(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m):
    sum = 0
    predict = forward_prop(X, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4)
    for i in range(int(m)):
        if (Y[i] == 1):
            sum += np.log(predict[0][i])
        else:
            sum += np.log(1 - predict[0][i])
    return (-(1 / m) * sum)

def gradient_check(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m):
    res_minus = cost_fct_sbs(X, Y, (thetas1 - 0.001), (thetas2 - 0.001), (thetas3 - 0.001), (thetas4 - 0.001), (bias1 - 0.001), (bias2 - 0.001), (bias3 - 0.001), (bias4 - 0.001), m)
    res_plus = cost_fct_sbs(X, Y, (thetas1 + 0.001), (thetas2 + 0.001), (thetas3 + 0.001), (thetas4 + 0.001), (bias1 + 0.001), (bias2 + 0.001), (bias3 + 0.001), (bias4 + 0.001), m)
    res_check = ((res_plus - res_minus) / (2 * 0.001))
    print(cost_fct_sbs(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m))
    print(res_check)
    #sys.exit()

def forward_prop_sbs(X, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4):
    l0 = np.reshape(X, (31, 1))
    l1 = tanh(thetas1.dot(l0) + bias1)
    l2 = tanh(thetas2.dot(l1) + bias2)
    l3 = tanh(thetas3.dot(l2) + bias3)
    return (sigmoid(thetas4.dot(l3) + bias4))

def make_regu(thetas1, thetas2, thetas3, thetas4, m):
    sum = 0
    for i in range(36):
        for j in range(31):
            sum += (thetas1[i][j])**2
    for i in range(36):
        for j in range(36):
            sum += (thetas2[i][j])**2
    for i in range(36):
        for j in range(36):
            sum += (thetas3[i][j])**2
    for i in range(1):
        for j in range(36):
            sum += (thetas4[i][j])**2
    return (((0.0001 / (2 * m) * sum)))

def cost_fct_sbs(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m):
    sum = 0
    for i in range(int(m)):
        if (Y[i] == 1):
            sum += np.log(forward_prop_sbs1(X[i], thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4))
        else:
            sum += np.log(1 - (forward_prop_sbs1(X[i], thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4)))
    regu = make_regu(thetas1, thetas2, thetas3, thetas4, m)
    return ((-(1 / m) * sum + regu))

def back_prop_sbs(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m):
    dthetas1 = np.zeros((36,31), dtype=float)
    dthetas2 = np.zeros((36,36), dtype=float)
    dthetas3 = np.zeros((36,36), dtype=float)
    dthetas4 = np.zeros((1,36), dtype=float)
    dbias1 = np.zeros((36,1), dtype=float)
    dbias2 = np.zeros((36,1), dtype=float)
    dbias3 = np.zeros((36,1), dtype=float)
    dbias4 = np.zeros((1,1), dtype=float)
    for i in range(int(m)):
        l1 = np.reshape(X[i], (31, 1))
        l2 = tanh((thetas1.dot(l1) + bias1))
        l3 = tanh((thetas2.dot(l2) + bias2))
        l4 = tanh((thetas3.dot(l3) + bias3))
        l5 = sigmoid((thetas4.dot(l4) + bias4))
        dlayer5 = l5 - Y[i]
        dlayer4 = ((thetas4.transpose().dot(dlayer5)) * (1 - (tanh(l4))**2))#(l2 * (1 - l2))) Dsigmoid
        dlayer3 = ((thetas3.transpose().dot(dlayer4)) * (1 - (tanh(l3))**2))#(l2 * (1 - l2))) Dsigmoid
        dlayer2 = ((thetas2.transpose().dot(dlayer3)) * (1 - (tanh(l2))**2))#(l1 * (1 - l1))) Dsigmoid
        dthetas1 += dlayer2.dot(l1.transpose())
        dthetas2 += dlayer3.dot(l2.transpose())
        dthetas3 += dlayer4.dot(l3.transpose())
        dthetas4 += dlayer5.dot(l4.transpose())
        dbias4 += np.sum(dlayer5, axis=1, keepdims=True)
        dbias3 += np.sum(dlayer4, axis=1, keepdims=True)
        dbias2 += np.sum(dlayer3, axis=1, keepdims=True)
        dbias1 += np.sum(dlayer2, axis=1, keepdims=True)
    thetas1 -= ((1 / m) * (0.543 * dthetas1) + (0.0001 * thetas1)) 
    thetas2 -= ((1 / m) * (0.543 * dthetas2) + (0.0001 * thetas2))
    thetas3 -= ((1 / m) * (0.543 * dthetas3) + (0.0001 * thetas3))
    thetas4 -= ((1 / m) * (0.543 * dthetas4) + (0.0001 * thetas4))
    bias1 -= ((1 / m) * (0.543 * dbias1))
    bias2 -= ((1 / m) * (0.543 * dbias2))
    bias3 -= ((1 / m) * (0.543 * dbias3))
    bias4 -= ((1 / m) * (0.543 * dbias4))

def make_regu1(thetas1, thetas2, m):
    sum = 0
    for i in range(13):
        for j in range(31):
            sum += (thetas1[i][j])**2
    for i in range(1):
        for j in range(13):
            sum += (thetas2[i][j])**2
    return (((0.0001 / (2 * m) * sum)))

def forward_prop_sbs1(X, thetas1, thetas2, bias1, bias2):
    l0 = np.reshape(X, (31, 1))
    l1 = tanh(thetas1.dot(l0) + bias1)
    return (float(sigmoid(thetas2.dot(l1) + bias2)))

def cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m):
    sum = 0
    for i in range(int(m)):
        if (Y[i] == 1):
            sum += np.log(forward_prop_sbs1(X[i], thetas1, thetas2, bias1, bias2))
        else:
            sum += np.log(1 - (forward_prop_sbs1(X[i], thetas1, thetas2, bias1, bias2)))
   # regu = make_regu1(thetas1, thetas2, m)
    return ((-(1 / m) * sum))# + regu))

def gradient_checkt(X, Y, thetas1, thetas2, bias1, bias2, dthetas1, dthetas2, dbias1, dbias2, ep, m):
    dthetas1 = np.reshape(dthetas1.ravel(), (403, 1))
    dthetas2 = np.reshape(dthetas2.ravel(), (13, 1))
    dbias1 = np.reshape(dbias1.ravel(), (13, 1))
    dbias2 = np.reshape(dbias2.ravel(), (1, 1))
    bigthetas = np.zeros((430, 1), dtype=float)
    bigdthetas = np.concatenate((dthetas1, dthetas2, dbias1, dbias2))
    k = 0
    for i in range(13):
        for j in range(31):
                tmp = thetas1[i][j]
                thetas1[i][j] += ep
                res_plus = cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
                thetas1[i][j] = tmp
                tmp = thetas1[i][j]
                thetas1[i][j] -= ep
                res_minus = cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
                thetas1[i][j] = tmp
                bigthetas[k][0] = ((res_plus - res_minus) / ( 2 * ep))
                k += 1
    for i in range(1):
        for j in range(13):
                tmp = thetas2[i][j]
                thetas2[i][j] += ep
                res_plus = cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
                thetas2[i][j] = tmp
                tmp = thetas2[i][j]
                thetas2[i][j] -= ep
                res_minus = cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
                thetas2[i][j] = tmp
                bigthetas[k][0] = ((res_plus - res_minus) / ( 2 * ep))
                k += 1
    for i in range(13):
        for j in range(1):
                tmp = bias1[i][j]
                bias1[i][j] += ep
                res_plus = cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
                bias1[i][j] = tmp
                tmp = bias1[i][j]
                bias1[i][j] -= ep
                res_minus = cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
                bias1[i][j] = tmp
                bigthetas[k][0] = ((res_plus - res_minus) / ( 2 * ep))
                k += 1
    for i in range(1):
        for j in range(1):
                tmp = bias2[i][j]
                print(bias2[i][j])
                bias2[i][j] += ep
                print(bias2[i][j])
                res_plus = cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
                bias2[i][j] = tmp
                print(bias2[i][j])
                tmp = bias2[i][j]
                bias2[i][j] -= ep
                print(bias2[i][j])
                res_minus = cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
                bias2[i][j] = tmp
                bigthetas[k][0] = ((res_plus - res_minus) / ( 2 * ep))
                k += 1
    sum1 = 0
    for i in range(430):
        sum1 += (bigthetas[i][0] - bigdthetas[i][0])**2
    sum1 = np.sqrt(sum1)
    sum2 = 0
    for i in range(430):
        sum2 += (bigthetas[i][0])**2
    sum2 = np.sqrt(sum2)
    sum3 = 0
    for i in range(430):
        sum3 += (bigdthetas[i][0])**2
    sum3 = np.sqrt(sum3)
    res = (sum1 / sum2 + sum3)
    print(res)
    sum1 = np.sqrt(np.sum(pow(bigthetas - bigdthetas, 2)))
    sum2 = np.sqrt(np.sum(pow(bigthetas, 2)))
    sum3 = np.sqrt(np.sum(pow(bigdthetas, 2)))
    res = (sum1 / sum2 + sum3)
    print(res)
    sys.exit()

def back_prop_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m):
    dthetas1 = np.zeros((13, 31), dtype=float)
    dthetas2 = np.zeros((1, 13), dtype=float)
    dbias1 = np.zeros((13, 1), dtype=float)
    dbias2 = np.zeros((1, 1), dtype=float)
    for i in range(int(m)):
        l1 = np.reshape(X[i], (31, 1))
        l2 = tanh((thetas1.dot(l1) + bias1))
        l3 = sigmoid((thetas2.dot(l2) + bias2))
        dlayer3 = l3 - Y[i]
        dlayer2 = ((thetas2.transpose().dot(dlayer3)) * (1 - (tanh(l2))**2))#(l1 * (1 - l1))) Dsigmoid
        dthetas1 += dlayer2.dot(l1.transpose())
        dthetas2 += dlayer3.dot(l2.transpose())
        dbias2 += np.sum(dlayer3, axis=1, keepdims=True)
        dbias1 += np.sum(dlayer2, axis=1, keepdims=True)
    dthetas1 = (((1 / m) * dthetas1))# + (0.0001 * thetas1))
    dthetas2 = (((1 / m) * dthetas2))# + (0.0001 * thetas2))
    dbias1 = ((1 / m) * dbias1)
    dbias2 = ((1 / m) * dbias2)
    thetas1 = (thetas1 - (0.009 * dthetas1))
    thetas2 = (thetas2 - (0.009 * dthetas2))
    bias1 = (bias1 - (0.009 * dbias1))
    bias2 = (bias2 - (0.009 * dbias2))
    gradient_checkt(X, Y, thetas1, thetas2, bias1, bias2, dthetas1, dthetas2, bias1, bias2, 0.0000001, m)
    return (thetas1, thetas2, bias1, bias2)

def gradient_check1(X, Y, thetas1, thetas2, bias1, bias2, m, ep):
    res_minus = cost_fct_sbs1(X, Y, (thetas1 - ep), (thetas2 - ep), (bias1 - ep), (bias2 - ep), m)
    res_plus = cost_fct_sbs1(X, Y, (thetas1 + ep), (thetas2 + ep), (bias1 + ep), (bias2 + ep), m)
    res_check = ((res_plus - res_minus) / (2 * ep))
    print(cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m))
    print(res_check)
    sys.exit()


def main():
    check_argv()
    data, Y = read_file()
    X = scaling_feat(data)
    m = X.shape[0]
    epsilon = 0.0000001
    thetas1 = (np.random.rand(13, X.shape[1]) * (2 * epsilon) - epsilon)
    #thetas2 = (np.random.rand(36, 36) * (2 * epsilon) - epsilon)
    thetas2 = (np.random.rand(1, 13) * (2 * epsilon) - epsilon)
    #thetas3 = (np.random.rand(36, 36) * (2 * epsilon) - epsilon)
    #thetas4 = (np.random.rand(1, 36) * (2 * epsilon) - epsilon)
    bias1 = (np.random.rand(13, 1) * (2 * epsilon) - epsilon)
    #bias2 = (np.random.rand(36, 1) * (2 * epsilon) - epsilon)
    bias2 = (np.random.rand(1, 1) * (2 * epsilon) - epsilon)
    #bias3 = (np.random.rand(36, 1) * (2 * epsilon) - epsilon)
    #bias4 = (np.random.rand(1, 1) * (2 * epsilon) - epsilon)
    #back_prop_sbs(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m)
    thetas1, thetas2, bias1, bias2 = back_prop_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
   # gradient_check1(X, Y, thetas1, thetas2, bias1, bias2, m, epsilon)
    #back_prop(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m)
   # gradient_check(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m)
    res_cost = []
    index = []
    for i in range(10):
        #back_prop_sbs(X, Y, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4, m)
        #print(thetas1[0],"\n")
        #print(thetas2,"\n")
        #print(bias1,"\n")
        print(bias2,"\n")
        thetas1, thetas2, bias1, bias2 = back_prop_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m)
        index.append(i)
        res_cost.append(cost_fct_sbs1(X, Y, thetas1, thetas2, bias1, bias2, m))
    plt.plot(index, res_cost, color='red')
    plt.show()
   # pred = forward_prop(X, thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4)
   # for i in range(int(m)):
   #     print(pred[0][i], Y[i])
   # for i in range(int(m)):
        #print(forward_prop_sbs(X[i], thetas1, thetas2, thetas3, thetas4, bias1, bias2, bias3, bias4), Y[i])
    #    print(forward_prop_sbs1(X[i], thetas1, thetas2, bias1, bias2), Y[i])

if (__name__ == "__main__"):
    main()
