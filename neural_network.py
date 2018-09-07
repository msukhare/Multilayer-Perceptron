# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neural_network.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/07 09:18:00 by msukhare          #+#    #+#              #
#    Updated: 2018/09/07 16:48:08 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

###initialisation des thetas
def create_w(nb_layers, nb_neurones, ep, nb_features):
    w = []
    w.append((np.random.rand(nb_neurones[0], nb_features) * ep))
    for i in range(1, int(nb_layers)):
        w.append((np.random.rand(nb_neurones[i], nb_neurones[i - 1]) * ep))
    return (w)

def create_bias(nb_layers, nb_neurones):
    bias = []
    for i in range(int(nb_layers)):
        bias.append(np.zeros((nb_neurones[i], 1), dtype=float))
    return (bias)
##Fin initialisation

class neural_network:

    def __init__(self):
        self._alpha = 0.02
        self._epoch = 1000
        self._epsilon_rand = 0.01
        self._nb_layer = 4
        self._nb_neurones = [36, 30, 15, 2]
        self._nb_features = 31
        self._w = create_w(self._nb_layer, self._nb_neurones, self._epsilon_rand, self._nb_features)
        self._bias = create_bias(self._nb_layer, self._nb_neurones)

##Debut de SET GET

###get de alpha, epoch, epsilon_for_rand, hidden_layer, nb_neurones_by_layers, w, bias

    @property
    def alpha(self):
        return (self._alpha)

    @property
    def epoch(self):
        return (self._epoch)

    @property
    def epsilon_rand(self):
        return (self._epsilon_rand)

    @property
    def nb_layer(self):
        return (self._nb_layer)

    @property
    def nb_neurones(self):
        return (self._nb_neurones)

    @property
    def nb_features(self):
        return (self._nb_features)

    @property
    def w(self):
        return (self._w)

    @property
    def bias(self):
        return (self._bias)
###set de alpha, epoch, epsilon_for_rand, hidden_layer, nb_neurones_by_layers, w, bias

    @alpha.setter
    def alpha(self, new):
        self._alpha = new

    @epoch.setter
    def epoch(self, new):
        self._epoch = new

    @epsilon_rand.setter
    def epsilon_rand(self, new):
        self._epsilon_rand = new

    @nb_layer.setter
    def nb_layer(self, new):
        self._nb_layer = new

    @nb_neurones.setter
    def nb_neurones(self, new):
        self._nb_neurones = new

    @nb_features.setter
    def nb_features(self, new):
        self._nb_features = new

    @w.setter
    def w(self, new):
        self._w = new

    @bias.setter
    def bias(self, new):
        self._bias = new
##FIN SET GET
