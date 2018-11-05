# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    use_neural_network.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/11/05 13:49:41 by msukhare          #+#    #+#              #
#    Updated: 2018/11/05 17:05:50 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import np as np
import pandas as pd
from neural_network import neural_network

def main():
    neural_n = neural_network()
    neural_n.read_architecture()
    layers = neural_n.forward_prop(X)
    pred = layers[neural_n.nb_layer].transpose()

if (__name__ == "__main__"):
    main()
