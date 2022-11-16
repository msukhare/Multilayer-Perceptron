import numpy as np

def l2_cost_fct(lambda_param, weights, m):
    return (lambda_param * np.sum(weights**2)) / (2 * m)

def l2_gradient(lambda_param, weights, m):
    return (lambda_param * weights.T) / m