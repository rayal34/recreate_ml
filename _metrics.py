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


def average_precision(actual, predict, k=None):
    if k is None:
        k = len(predict)

    predict = predict[:k]

    num_tp = 0
    numerator = 0
    for i, p in enumerate(predict, 1):
        relevance = (p in actual) * 1
        num_tp += relevance
        precision_at_k = num_tp / i
        numerator += precision_at_k * relevance

    return numerator / k


def mean_average_precision(actuals, predicts, k=None):
    return np.mean([average_precision(a, p, k) for a, p in zip(actuals, predicts)])


def roc_auc_score(y_true, y_score):
    pass