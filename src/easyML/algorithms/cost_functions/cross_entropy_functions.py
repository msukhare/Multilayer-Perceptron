import numpy as np

def binary_cross_entropy(Y, Y_pred):
    return (1 / Y.shape[0]) * (-Y.transpose().dot(np.log(Y_pred)) -\
        (1 - Y).transpose().dot(np.log(1 - Y_pred)))

def categorical_cross_entropy(Y, Y_pred):
    return np.sum(-Y * np.log(Y_pred)) / Y.shape[0]