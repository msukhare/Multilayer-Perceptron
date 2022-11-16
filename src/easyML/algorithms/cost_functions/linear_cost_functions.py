import numpy as np

def mse(Y_pred, Y):
    return np.sum((Y - Y_pred)**2, axis=0) / Y.shape[0]

def rmse(Y_pred, Y):
    return np.sqrt(mse(Y_pred, Y))

def mae(Y_pred, Y):
    return np.sum(np.abs(Y - Y_pred), axis=0) / Y.shape[0]
