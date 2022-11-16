import numpy as np

def softmax(z):
    expo = np.exp(z)
    return (expo / (np.sum(expo, axis=0)))