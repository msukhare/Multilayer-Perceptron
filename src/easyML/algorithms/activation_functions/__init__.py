from .linear import linear_hypothesis
from .sigmoid_fct import sigmoid
from .softmax_fct import softmax
from .tanh_fct import tanh,\
                        dtanh
from .leaky_relu_fct import leaky_relu,\
                            dleaky_relu
from .relu_fct import relu,\
                        drelu

ACTIVATION_FUNCTION = {'tanh': tanh,\
                    'leaky_relu': leaky_relu,\
                    'relu': relu,\
                    'sigmoid': sigmoid,\
                    'softmax': softmax}
DERIVATIVE_FUNCTION = {'tanh': dtanh,\
                    'leaky_relu': dleaky_relu,\
                    'relu': drelu}