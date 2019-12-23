import numpy as np


def accuracy(y, y_pred):
    return (y == y_pred).mean()


def roc_curve(y_true, y_score):
    threshold = sorted(y_score, reverse=True)
    fpr, tpr = np.zeros(len(threshold)), np.zeros(len(threshold))

    for i, t in enumerate(threshold):
        predicted_positives = y_score >= t
        tp = (predicted_positives == 1) and (y_true == 1)
        fp = (predicted_positives == 1) and (y_true == 0)
        tn = (predicted_positives == 0) and (y_true == 0)
        fn = (predicted_positives == 0) and (y_true == 1)
        tpr[i] = tp / (tp + fn)
        fpr[i] = fp / (tn + fp)

    return fpr, tpr, threshold


def roc_auc_score(y_true, y_score):
    pass