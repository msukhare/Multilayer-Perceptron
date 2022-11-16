# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    activation_function.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/10 16:34:09 by msukhare          #+#    #+#              #
#    Updated: 2018/09/10 16:34:26 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

def tanh(z):
    return (((np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))))

def dtanh(z):
    return ((1 - (tanh(z)**2)))

def relu(z):
    z[z < 0] = 0
    return (z)

def drelu(z):
    return (1 * (z >= 0))

def leaky_relu(z):
    z[z < 0] *= 0.01
    return (z)

def dleaky_relu(z):
    z[z >= 0] = 1
    z[z < 0] = 0.01
    return (z)

def softmax(z):
    return (np.exp(z) / (np.sum(np.exp(z), axis=0)))

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def activate_layer(z, activation_function):
    if (activation_function == 10):
        return (tanh(z))
    elif (activation_function == 11):
        return (relu(z))
    elif (activation_function == 12):
        return (leaky_relu(z))
    elif (activation_function == 2):
        return (softmax(z))
    else:
        return (sigmoid(z))

def deri_func(z, activation_function):
    if (activation_function == 10):
        return (dtanh(z))
    elif (activation_function == 11):
        return (drelu(z))
    elif (activation_function == 12):
        return (dleaky_relu(z))
    return (0)
