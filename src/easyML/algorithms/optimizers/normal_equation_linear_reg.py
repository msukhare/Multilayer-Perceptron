import numpy as np

def normal_equation(X, Y):
    X_T = np.transpose(X)
    return np.linalg.inv(X_T.dot(X)).dot(X_T).dot(Y)
