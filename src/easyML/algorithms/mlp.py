import configparser
import numpy as np

from ..data_managment import split_data
from .activation_functions import ACTIVATION_FUNCTION,\
                                    DERIVATIVE_FUNCTION
from .cost_functions import binary_cross_entropy,\
                            categorical_cross_entropy

class MLP:

    def __init__(self,\
                config_file_path,\
                show_arch=False,\
                type_of_initialization='xavier',\
                epochs=100,\
                batch_size=None,\
                lr=0.01,\
                validation_fraction=0.10,\
                l2=False,\
                lambda_value=0.01,\
                early_stopping=False,\
                n_epochs_no_change=5,\
                tol=1e-3,\
                accuracy=False,\
                precision=False,\
                recall=False,\
                f1_score=False):
        self.config_file_path = config_file_path
        self.show_arch = show_arch
        self.type_of_initialization = type_of_initialization
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.l2 = l2
        self.lambda_value = lambda_value
        self.early_stopping = early_stopping
        self.n_epochs_no_change = n_epochs_no_change
        self.tol = tol
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.architecture = None
        self.loss = None
        self.classes = None

    def init_layer(self, section, config, input_size):
        activation_fct = None
        batch_norm = False
        bias = True
        if config.has_option(section, 'output_size') is False:
            raise Exception("missing output_size in section %s" %section)
        output_size = int(config.get(section, 'output_size'))
        if config.has_option(section, 'bias') is True:
            bias = config.get(section, 'bias')
        if config.has_option(section, 'activation_function') is True:
            activation_fct = config.get(section, 'activation_function')
            if section == "OUTPUT_LAYER":
                if activation_fct == "softmax":
                    self.loss = categorical_cross_entropy
                else:
                    self.loss = binary_cross_entropy
                    output_size = 1
        elif section == 'OUTPUT_LAYER':
            print("WARNING: outputlayer must have an activation function sigmoid or softmax, bad behavior will result")
        if config.has_option(section, 'batch_norm') is True:
            batch_norm = config.get(section, 'batch_norm')
        return {'weights': np.random.rand(output_size, input_size) * 0.1,\
                'bias': np.zeros((output_size, 1)) if bias is True else None,\
                'activation_function': activation_fct,\
                'batch_norm': batch_norm},\
                output_size

    def init_mlp(self, input_size):
        architecture = {}
        config = configparser.ConfigParser()
        config.read(self.config_file_path)
        for section in config.sections():
            if section == 'INPUT_LAYER' and\
                'INPUT_LAYER' not in architecture.keys():
                architecture['INPUT_LAYER'], input_size = self.init_layer(section, config, input_size)
            elif section == 'OUTPUT_LAYER' and\
                'INPUT_LAYER' in architecture.keys():
                architecture['OUTPUT_LAYER'], input_size = self.init_layer(section, config, input_size)
                break
            elif 'INPUT_LAYER' in architecture.keys() and\
                'OUTPUT_LAYER' not in architecture.keys():
                architecture[section], input_size = self.init_layer(section, config, input_size)
            else:
                raise Exception("something is wrong with cfg file %s" %self.config_file_path)
        if 'INPUT_LAYER' not in architecture.keys() or\
            'OUTPUT_LAYER' not in architecture.keys():
            raise Exception("missing INPUT_LAYER or OUTPUT_LAYER")
        if self.show_arch is True:
            print(architecture)
        return architecture

    def transform_y(self, Y):
        Y = np.asarray(Y)
        labels = np.sort(np.unique(Y))
        if labels.shape[0] <= 2 and\
            self.architecture["OUTPUT_LAYER"]['activation_function'] == "sigmoid":
            new_Y = np.reshape(Y, (Y.shape[0], 1))
        else:
            new_Y = np.zeros((Y.shape[0], labels.shape[0]))
            for i in range(Y.shape[0]):
                new_Y[i][np.where(labels == Y[i])[0]] = 1
        return labels, new_Y
    
    def forward_prop(self, X):
        ret = []
        ret.append(X.transpose())
        for i in range(int(self.nb_layer)):
            ret.append(activate_layer((self.w[i].dot(ret[i]) + self.bias[i]),\
                    self.activate_func[i]))
        return (ret)
    
    def forward(self, X):
        output = X.T
        for layer in self.architecture.keys():
            output = self.architecture[layer]['weights'].dot(output)
            if self.architecture[layer]['bias'] is not None:
                output += self.architecture[layer]['bias']
            if self.architecture[layer]['activation_function'] is not None:
                output = ACTIVATION_FUNCTION[self.architecture[layer]['activation_function']](output)
            self.architecture[layer]['layer'] = output
        return output

    def gradient_descent(self, dw, dbias):
        j = (self.nb_layer - 1)
        for i in range(int(self.nb_layer)):
            self.w[j] = (self.w[j] - (self.alpha * dw[i]))
            j -= 1
        j = (self.nb_layer - 1)
        for i in range(int(self.nb_layer)):
            self.bias[j] = (self.bias[j] - (self.alpha * dbias[i]))
            j -= 1

    def get_derivate_layer(self, ls, Y):
        dl = []
        dl.append((ls[self.nb_layer] - Y.transpose()))
        j = 0
        for i in range(int(self.nb_layer - 1), 0, -1):
            dl.append(((self.w[i].transpose().dot(dl[j])) * \
                deri_func((self.w[i - 1].dot(ls[i - 1]) + self.bias[i - 1]),
                    self.activate_func[i - 1])))
            j += 1
        return (dl)

    def get_derivate_w(self, dl, ls, m):
        dw = []
        j = (self.nb_layer - 1)
        for i in range(int(self.nb_layer)):
            dw.append(((1 / m) * (dl[i].dot(ls[j].transpose()))))
            j -= 1
        return (dw)

    def get_derivate_bias(self, dl, m):
        dbias = []
        for i in range(int(self.nb_layer)):
            dbias.append(((1 / m) * (np.sum(dl[i], axis=1, keepdims=True))))
        return (dbias)

    ###Backpropagation and update thetas and bias with gradient descent
    def back_propagation(self, X, Y, m):
        ls = self.forward_prop(X)
        dl = self.get_derivate_layer(ls, Y)
        return (self.get_derivate_w(dl, ls, m), self.get_derivate_bias(dl, m))
        #dw, dbias = self.back_propagation(X_train, Y_train, m)
        #self.gradient_descent(dw, dbias)

    def backward(self, X, Y):
        layers = list(reversed(self.architecture.keys()))
        dlayer = self.architecture["OUTPUT_LAYER"]['layer'] - Y.transpose()
        for i, layer in enumerate(layers):
            dlayer = self.architecture[layer]['weights'].transpose().dot(dlayer)
            if self.architecture[layer]['activation_function'] is not None:
                before = self.architecture[layers[i + 1]]['weights'].dot(self.architecture[layers[i + 2]]['layer'] if i + 2 < len(layers) else X.T)
                before += self.architecture[layers[i + 1]]['bias'] if self.architecture[layers[i + 1]]['bias'] is not None else 0
            if self.architecture[layer]['activation_function'] is not None:
                before = DERIVATIVE_FUNCTION[self.architecture[layers[i + 1]]['activation_function']](before)
                dlayer = dlayer * DERIVATIVE_FUNCTION[self.architecture[layers[i + 1]]['activation_function']](before)

    def predict_proba(self, X):
        pass
    
    def predict(self, X):
        pass

    def get_batch(self, X, Y, batch_size):
        i = 0
        while (i < X.shape[0]):
            yield X[i: i + batch_size], Y[i: i + batch_size]
            i += batch_size
        if i < X.shape[0]:
            yield X[i:], Y[i:]

    def evaluate_training(self, X, Y):
        global_loss = 0
        batch_size = self.batch_size
        confusion_matrix = np.zeros((self.classes.shape[0], self.classes.shape[0]))
        if batch_size is None or\
            batch_size <= 0:
            batch_size = X.shape[0]
        for iter_, (X_batch, Y_batch) in enumerate(self.get_batch(X, Y, batch_size)):
            predicted_y, loss = self.kernel.eval_on_batch(X_batch,\
                                                        Y_batch,\
                                                        self.classes,\
                                                        self.weights,\
                                                        self.l2,\
                                                        self.lambda_value)
            global_loss += loss
            for i, ele in enumerate(predicted_y):
                confusion_matrix[np.argmax(Y_batch[i])][ele] += 1
        return confusion_matrix, global_loss / (iter_ + 1)

    def fit(self, X, Y):
        batch_size_train = self.batch_size
        self.architecture = self.init_mlp(X.shape[1])
        self.classes, Y = self.transform_y(Y)
        X_train, X_val, Y_train, Y_val = split_data(X, Y, 1 - self.validation_fraction)
        #if self.early_stopping is True:
        #    confusion_matrix, best_loss_val = self.evaluate_training(X_val, Y_val)
        if self.batch_size is None or\
            self.batch_size <= 0:
            batch_size_train = X.shape[0]
        nb_epochs_waiting = 0
        for epoch in range(self.epochs):
            global_loss = 0
            training_process = ""
            for iter_, (X_batch, Y_batch) in enumerate(self.get_batch(X_train, Y_train, batch_size_train)):
                outputs = self.forward(X_batch)
                #print(outputs.shape)
                #print(Y_batch.shape)
                loss = self.loss(Y_batch.T, outputs)
                self.backward(X_batch, Y_batch)
                global_loss += loss
            training_process += "epoch %d/%d epochs; loss train is equal to %f" %(epoch, self.epochs, global_loss / (iter_ + 1))
            #confusion_matrix, loss_val = self.evaluate_training(X_val, Y_val)
            #training_process += "; loss val is equal to %f" %(loss_val)
            #if self.accuracy is True:
            #    training_process += "; val accuracy is equal to %f" %(accuracy_score(confusion_matrix))
            #if self.precision is True:
            #    training_process += "; val precision is equal to %f" %(precision_score(confusion_matrix,\
            #                                                                            self.average))
            #if self.recall is True:
            #    training_process += "; val recall is equal to %f" %(recall_score(confusion_matrix,\
            #                                                                    self.average))
            #if self.f1_score is True:
            #    training_process += "; val f1_score is equal to %f" %(f1_score(confusion_matrix,\
            #                                                                    self.average))
            #if self.early_stopping is True:
            #    if loss_val > best_loss_val - self.tol:
            #        nb_epochs_waiting += 1
            #    else:
            #        nb_epochs_waiting = 0
            #    if nb_epochs_waiting >= self.n_epochs_no_change:
            #        break
            #    best_loss_val = min(loss_val, best_loss_val)
            print(training_process)

    def save_weights(self, path_where_save_weights, params_to_save, labels):
        pass

    def load_weights(self, path_to_weights):
        pass