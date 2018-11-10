# Multilayer-Perceptron

## About Multilayer-Perceptron:

* Multilayer-Perceptron is a machine learning project. The goal of this project is to do an Artificial Neural Network from scratch.

* Multilayer-Perceptron is composed of two scripts, `train_neural_network.py`, `use_neural_network.py` and 2 classes `NeuralNetwork`, `metrics_for_binary_classification.py`.

### About `train_neural_network.py`:

* `train_neural_network.py` creates an instance of class `NeuralNetwork` and calls `train_thetas` method to train weight.

* `train_neural_network.py` creates an instance of class 'metrics_for_binary_classification` and calls `confusion_matrix_softmax` to get metrics.

* This script writes architecture (number of layers, number of features in data, number of neurons per layers, activation function for each layers) of neural_network in `architecture_of_mlp.txt` and weights in `weight.npy`(serialized matrix).

### About `use_neural_network.py`:

* `use_neural_network.py` reads `architecture_of_mlp.txt` and `weight.npy` to restore the network to do predictions on new data.

* `use_neural_network.py` writes predictions in `predictions_for_testset.csv`

### About `NeuralNetwork` class:

* train_thetas method use gradient descent to update weights.

* I use cross entropy as cost function.

* Softmax is used for the output layer.

* [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) is used to train network.

### About `MetricsForBinaryClassification` class:

* The metrics given by confusion_matrix_soft_max method are accuracy, precision, recall, f1, classification error, false alarm rate, miss rate and kappa cohen.

### About data:

* `data.csv` shape are 570 exemples, 31 features.

* `test.csv` is the 42 last exemples of `data.csv`

### about `activation_function.py`:

* Contains softmax, tanh, sigmoid, relu, leakyrelu.

## What you need to make Multilayer-Perceptron work

* python >= 3.0

* [pandas](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html)

* [numpy](http://www.numpy.org/)

* [matplotlib](https://matplotlib.org/)

## Usage:

* `python3.0 train_neural_network.py data.csv`.

* `python3.0 use_neural_network.py FileToPredict.csv`. FileToPredict.cvs must have the same number of features as data for train.
