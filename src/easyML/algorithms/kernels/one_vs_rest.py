import numpy as np

from ..activation_functions import sigmoid
from ..cost_functions import binary_cross_entropy
from ..optimizers import compute_dweights,\
                        gradient_descent
from ..regularization import l2_cost_fct,\
                            l2_gradient

class OVR:

    def transform_y(self, Y):
        labels = np.sort(np.unique(np.asarray(Y)))
        new_Y = np.zeros((Y.shape[0], labels.shape[0]))
        for i in range(Y.shape[0]):
            new_Y[i][np.where(labels == Y[i])[0]] = 1
        return labels, new_Y

    def init_weights(self, weights, classes, X):
        if weights is None or weights.shape[1] != X.shape[1] or\
            weights.shape[0] != classes.shape[0]:
            return np.zeros((classes.shape[0], X.shape[1]))
        return weights

    def predict_proba(self, X, classes, weights):
        predicted_proba = []
        for index, lab in enumerate(classes):
            predicted_proba.append(np.expand_dims(sigmoid(-X.dot(weights[index])), axis=1))
        return np.concatenate(predicted_proba, axis=1)

    def predict(self, X, classes, weights):
        predicted_class = []
        predicted_proba = self.predict_proba(X, classes, weights)
        for prediction in predicted_proba:
            predicted_class.append(classes[np.argmax(prediction)])
        return predicted_class

    def infer_on_batch(self, X, Y, classes, weights, lr, l2, lambda_value):
        global_loss = 0
        for index, lab in enumerate(classes):
            forward = sigmoid(-X.dot(weights[index]))
            loss = binary_cross_entropy(Y[:, index: index + 1], forward)[0]
            DW = compute_dweights(X, forward, Y[:, index: index + 1].reshape(Y.shape[0]))
            weights[index] = gradient_descent(weights[index], DW, lr)
            if l2 is True:
                loss += l2_cost_fct(lambda_value, weights[index, 1: ], X.shape[0])
                weights[index, 1:] += l2_gradient(lambda_value, weights[index, 1: ], X.shape[0])
            global_loss += loss
        return weights, global_loss / classes.shape[0]

    def eval_on_batch(self, X, Y, classes, weights, l2, lambda_value):
        global_loss = 0
        y_pred = []
        for index, lab in enumerate(classes):
            forward = sigmoid(-X.dot(weights[index]))
            y_pred.append(np.expand_dims(forward, axis=1))
            loss = binary_cross_entropy(Y[:, index: index + 1], forward)[0]
            if l2 is True:
                loss += l2_cost_fct(lambda_value, weights[index][1: ], X.shape[0])
            global_loss += loss
        return np.argmax(np.concatenate(y_pred, axis=1), axis=1), global_loss / classes.shape[0]