import numpy as np

def standardization(X, params, epsilon=1e-8):
    return (X - params['mean']) / (params['std'] + epsilon)

def rescaling(X, params, epsilon=1e-8):
    return (X - params['min']) / ((params['max'] - params['min']) + epsilon)

def normalization(X, params, epsilon=1e-8):
    return (X - params['mean']) / ((params['max'] - params['min']) + epsilon)

def scaling_features(X, params=None, features_scaling_method=None):
    features_scaling_methods = {'standardization': standardization,\
                            'rescaling': rescaling,\
                            'normalization': normalization}
    if params is None:
        params = {'method_scaling': features_scaling_method,\
                'mean': np.mean(X, axis=0),\
                'std': np.std(X, axis=0),\
                'max': np.max(X, axis=0),\
                'min': np.min(X, axis=0)}
    return features_scaling_methods[params['method_scaling']](X, params), params
