import numpy as np

def accuracy_score(confusion_matrix):
    return np.sum(np.diag(confusion_matrix)) / confusion_matrix.sum(axis=None)

def precision_score(confusion_matrix, average='macro'):
    TP, FP = [], []
    for class_i in range(confusion_matrix.shape[0]):
        TP.append(confusion_matrix[class_i][class_i])
        FP.append(np.abs(np.sum(confusion_matrix[:, class_i: class_i + 1]) - confusion_matrix[class_i][class_i]))
    if average == "macro":
        precision = 0
        for class_i in range(confusion_matrix.shape[0]):
            precision += TP[class_i] / (TP[class_i] + FP[class_i])
        return precision / confusion_matrix.shape[0]
    return np.sum(TP) / (np.sum(TP) + np.sum(FP))

def recall_score(confusion_matrix, average='macro'):
    TP, FN = [], []
    for class_i in range(confusion_matrix.shape[0]):
        TP.append(confusion_matrix[class_i][class_i])
        FN.append(np.abs(np.sum(confusion_matrix[class_i: class_i + 1, :]) - confusion_matrix[class_i][class_i]))
    if average == "macro":
        recall = 0
        for class_i in range(confusion_matrix.shape[0]):
            recall += TP[class_i] / (TP[class_i] + FN[class_i])
        return recall / confusion_matrix.shape[0]
    return np.sum(TP) / (np.sum(TP) + np.sum(FN))

def f1_score(confusion_matrix, average='macro'):
    precision = precision_score(confusion_matrix, average)
    recall = recall_score(confusion_matrix, average)
    return (2 * precision * recall) / (precision + recall)