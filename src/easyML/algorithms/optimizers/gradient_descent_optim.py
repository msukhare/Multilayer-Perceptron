import numpy as np

def compute_dweights(X, Y_pred, Y):
    return X.T.dot(Y_pred - Y) / X.shape[0]

def gradient_descent(weights, dweights, lr):
    return weights - (lr * dweights)