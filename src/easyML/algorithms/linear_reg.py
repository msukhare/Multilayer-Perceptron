import numpy as np
import pickle
import os

from .optimizers import gradient_descent, compute_dweights, normal_equation
from .cost_functions import linear_cost_function_methods
from .activation_functions import linear_hypothesis
from ..data_managment import split_data

class LinearReg:

    def __init__(self,\
                optimizer='SGD',\
                lr=0.1,\
                cost_funct='MSE',\
                epochs=100,\
                early_stopping=False,\
                val_frac=0.10,\
                n_epochs_no_change=5,\
                tol=1e-3,\
                show_training=False):
        self.weights = None
        self.optimizer = optimizer
        self.lr = lr
        self.name_cost_fct = cost_funct
        self.cost_funct = linear_cost_function_methods[cost_funct]
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.val_frac = val_frac
        self.n_epochs_no_change = n_epochs_no_change
        self.tol = tol
        self.show_training = show_training

    def _train_weights(self, X, Y):
        if self.early_stopping is True:
            X, X_val, Y, Y_val = split_data(X, Y, 1 - self.val_frac)
            best_loss_val = self.cost_funct(linear_hypothesis(X_val, self.weights), Y_val)
        nb_epochs_waiting = 0
        for i in range(self.epochs):
            training_process = "%d/%d " %(i, self.epochs)
            self.weights = gradient_descent(self.weights,\
                                    compute_dweights(X, linear_hypothesis(X, self.weights), Y),\
                                    self.lr)
            train_loss = self.cost_funct(linear_hypothesis(X, self.weights), Y)
            training_process += '%f %s train set' %(train_loss, self.name_cost_fct)
            if self.early_stopping is True:
                loss_val = self.cost_funct(linear_hypothesis(X_val, self.weights), Y_val)
                training_process += '; %f %s val set' %(loss_val, self.name_cost_fct)
            if self.show_training:
                print(training_process)
            if self.early_stopping is True:
                if loss_val > best_loss_val - self.tol:
                    nb_epochs_waiting += 1
                else:
                    nb_epochs_waiting = 0
                if nb_epochs_waiting >= self.n_epochs_no_change:
                    return
                best_loss_val = min(loss_val, best_loss_val)

    def fit(self, X, Y):
        if X.shape[0] != Y.shape[0]:
            raise Exception("Number samples in X must be equal to number sample in Y")
        X = np.concatenate((np.ones((X.shape[0], 1), dtype=float), X), axis=1)
        if self.optimizer == "normal_equation":
            self.weights = normal_equation(X, Y)
        else:
            if self.epochs <= 0:
                raise Exception("Number of epochs must be superior or equal to 1")
            if self.early_stopping is True and self.val_frac <= 0 or self.val_frac >= 1:
                raise Exception("val_frac must be superior to 0 and inferior to 1")
            if self.weights is None or self.weights.shape[0] != X.shape[1]:
                self.weights = np.zeros(X.shape[1], dtype=float)
            self._train_weights(X, Y)

    def predict(self, X):
        if self.weights is None:
            raise Exception("Weights must be load or trained at first")
        X = np.concatenate((np.ones((X.shape[0], 1), dtype=float), X), axis=1)
        if self.weights.shape[0] != X.shape[1]:
            raise Exception('Number of weights (%d) must be equal to number features (%d)'
                    %(self.weights.shape[0], X.shape[1]))
        return linear_hypothesis(X, self.weights)

    def features_importance(self, columns):
        #TO DO
        pass

    def eval(self, X, Y):
        print("%s is equal to %f on test set" %(self.name_cost_fct,\
                                    self.cost_funct(self.predict(X), Y)))

    def save_weights(self, file_name_weights, pipeline=None):
        if file_name_weights is not None:
            to_save = {'info_about_pipeline': pipeline,\
                        'weights': self.weights}
            with open(file_name_weights + '.pth', 'wb') as fd:
                pickle.dump(to_save, fd)
            print("saving of weights is done in %s.pth" %file_name_weights)

    def load_weights(self, file_name_weights):
        if os.path.exists(file_name_weights) is False:
            raise Exception("%s doesn't exist" %file_name_weights)
        if '.pth' not in file_name_weights:
            raise Exception("Extension of file must be .pth in %s" %file_name_weights)
        with open(file_name_weights, 'rb') as fd:
            pipeline = pickle.load(fd)
        if 'info_about_pipeline' not in pipeline.keys() or\
                'weights' not in pipeline.keys():
            raise Exception('Missing information in %s'%file_name_weights)
        self.weights = pipeline['weights']
        return pipeline['info_about_pipeline']
