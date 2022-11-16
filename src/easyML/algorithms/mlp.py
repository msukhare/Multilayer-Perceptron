import configparser
import numpy as np

from ..data_managment import split_data

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
        elif section == 'OUTPUT_LAYER':
            print("WARNING: outputlayer must have an activation function sigmoid or softmax, bad behavior will result")
        if config.has_option(section, 'batch_norm') is True:
            batch_norm = config.get(section, 'batch_norm')
        return {'weights': np.random.rand(input_size, output_size) * 0.1,\
                'bias': np.zeros((1, output_size)) if bias is True else None,\
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
        pass

    def forward(self, X):
        output = X
        for layer in self.architecture.keys():
            output = output.dot(self.architecture[layer]['weights'])
            if self.architecture[layer]['bias'] is not None:
                output += self.architecture[layer]['bias']
            #if layer['activation_function'] is not None:
            #    ACTIVATION_FUNCTION[layer['activation_function']](output)
            self.architecture[layer]['layer'] = output
        return output

    def backward(self):
        pass

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
        #self.classes, Y = self.transform_y(Y)
        self.architecture = self.init_mlp(X.shape[1])
        X_train, X_val, Y_train, Y_val = split_data(X, Y, self.validation_fraction)
        #if self.early_stopping is True:
        #    confusion_matrix, best_loss_val = self.evaluate_training(X_val, Y_val)
        if self.batch_size is None or\
            self.batch_size <= 0:
            batch_size_train = X.shape[0]
        nb_epochs_waiting = 0
        print(X_train.shape)
        for epoch in range(self.epochs):
            global_loss = 0
            training_process = ""
            for iter_, (X_batch, Y_batch) in enumerate(self.get_batch(X_train, Y_train, batch_size_train)):
                outputs = self.forward(X_batch)
                print(outputs.shape)
                print(outputs)
                loss = self.cost_function(output, Y_batch)
                #self.backward()
                #global_loss += loss
            #training_process += "epoch %d/%d epochs; loss train is equal to %f" %(epoch, self.epochs, global_loss / (iter_ + 1))
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
            #print(training_process)

    def save_weights(self, path_where_save_weights, params_to_save, labels):
        pass

    def load_weights(self, path_to_weights):
        pass