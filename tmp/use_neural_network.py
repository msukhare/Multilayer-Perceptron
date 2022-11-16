# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    use_neural_network.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/11/05 13:49:41 by msukhare          #+#    #+#              #
#    Updated: 2018/11/10 03:08:51 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
import sys
import csv

def scaling_feat(data):
    X = data.iloc[:, 0:]
    X = np.array(X.values, dtype=float)
    desc = data.describe()
    j = 0
    for key in data:
        for i in range(int(desc[key]['count'])):
            X[i][j] = (X[i][j] - desc[key]['mean']) / desc[key]['std']#(desc[key]['max'] - desc[key]['min'])
        j += 1
    return (X)

def read_file():
    try:
        data = pd.read_csv(sys.argv[1])
    except:
        sys.exit("file doesn't exist")
    for key in data:
        data.fillna(value={key: data[key].mean()}, inplace=True)
    return (scaling_feat(data))

def write_predictions(pred):
    try:
        file = csv.writer(open("predictions_for_testset.csv", "w"), delimiter=',',\
                quoting=csv.QUOTE_MINIMAL)
    except:
        sys.exit("fail to create predictions_for_testset.csv")
    file.writerow(["place patients in testset", "Benin_Malin"])
    for i in range(pred.shape[0]):
        if (pred[i][0] > pred[i][1]):
            tum = 'B'
        else:
            tum = 'M'
        file.writerow([i, tum])

def main():
    if (len(sys.argv) <= 1):
        sys.exit("need more file")
    if (len(sys.argv) >= 3):
        sys.exit("too much file")
    X = read_file()
    neural_n = NeuralNetwork()
    neural_n.read_architecture()
    if (X.shape[1] < neural_n.nb_features):
        sys.exit("need more features")
    if (X.shape[1] > neural_n.nb_features):
        sys.exit("too much features")
    layers = neural_n.forward_prop(X)
    pred = layers[neural_n.nb_layer].transpose()
    write_predictions(pred)

if (__name__ == "__main__"):
    main()
