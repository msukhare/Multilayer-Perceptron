import numpy as np

from ..activation_functions import softmax
from ..cost_functions import categorical_cross_entropy
from ..optimizers import compute_dweights,\
                        gradient_descent
from ..regularization import l2_cost_fct,\
                            l2_gradient
class Multinomial:

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
            predicted_proba.append(np.expand_dims(softmax(weights.dot(X.T)).T, axis=1))
        return np.concatenate(predicted_proba, axis=1)

    def predict(self, X, classes, weights):
        predicted_class = []
        predicted_proba = self.predict_proba(X, classes, weights)
        for prediction in predicted_proba:
            predicted_class.append(classes[np.argmax(prediction)])
        return predicted_class

    def infer_on_batch(self, X, Y, classes, weights, lr, l2, lambda_value):
        forward = softmax(weights.dot(X.T)).T
        loss = categorical_cross_entropy(Y, forward)
        DW = compute_dweights(X, forward, Y).T
        weights = gradient_descent(weights, DW, lr)
        if l2 is True:
            loss += l2_cost_fct(lambda_value, weights[:, 1: ], X.shape[0])
            weights[:, 1: ] += l2_gradient(lambda_value, weights[:, 1: ], X.shape[0]).T
        return weights, loss

    def eval_on_batch(self, X, Y, classes, weights, l2, lambda_value):
        y_pred = softmax(weights.dot(X.T)).T
        global_loss = categorical_cross_entropy(Y, y_pred)
        if l2 is True:
            global_loss += l2_cost_fct(lambda_value, weights[:, 1:], X.shape[0])
        return np.argmax(y_pred, axis=1), global_loss