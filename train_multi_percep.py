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

def forward_prop(X, thetas1, thetas2, thetas3, layer):
    l2 = sigmoid(np.dot(thetas1, X))
    if (layer == 2):
        return (l2)
    l3 = sigmoid(np.dot(thetas2, l2))
    if (layer == 3):
        return (l3)
    return (sigmoid(np.dot(thetas3, l3)))

def get_summ(X, Y, thetas1, thetas2, thetas3, m):
    summ = 0
    for i in range(int(m)):
        if (Y[i] == 1):
            summ += np.log(forward_prop(X[i], thetas1, thetas2, thetas3, 4))
        else:
            summ += np.log(1 - forward_prop(X[i], thetas1, thetas2, thetas3, 4))
    return (summ)

def cost_fct(X, Y, thetas1, thetas2, thetas3):
    m = X.shape[0]
    return (-(1 / m) * get_summ(X, Y, thetas1, thetas2, thetas3, m))

def back_prop(X, Y, thetas1, thetas2, thetas3):
    for i in range(int(X.shape[0])):
        l4 = forward_prop(X[i], thetas1, thetas2, thetas3, 4)
        error4 = l4 - Y[i]
        l3 = forward_prop(X[i], thetas1, thetas2, thetas3, 3)
        error3 = (thetas3.transpose() * error4).dot(l3.dot(1 - l3))
        l2 = forward_prop(X[i], thetas1, thetas2, thetas3, 2)
        error2 = (thetas2.transpose() * error3).dot(l2.dot(1 - l2))
        tmp = 0 + error2.dot(X[i].transpose())
        print(tmp, tmp.shape[0])
        sys.exit()

def main():
    check_argv()
    data, Y = read_file()
    X = scaling_feat(data)
    epsilon = 0.01
    thetas1 = (np.random.rand(36, X.shape[1]) * (2 * epsilon) - epsilon)
    thetas2 = (np.random.rand(36, 36) * (2 * epsilon) - epsilon)
    thetas3 = (np.random.rand(1, 36) * (2 * epsilon) - epsilon)
    cost_fct(X, Y, thetas1, thetas2, thetas3)
    back_prop(X, Y, thetas1, thetas2, thetas3)

if (__name__ == "__main__"):
    main()
