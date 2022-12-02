import numpy as np

def tanh(z):
    expo_pos = np.exp(z)
    expo_neg = np.exp(-z)
    return (((expo_pos - expo_neg) / (expo_pos + expo_neg)))

def dtanh(z):
    return ((1 - (tanh(z)**2)))