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


def precision(y_true, y_pred):
    tp_sum = ((y_true == 1) and (y_pred == 1)).sum()
    fp_sum = ((y_true == 0) and (y_pred == 1)).sum()
    return tp_sum / (tp_sum + fp_sum)


def recall(y_true, y_pred):
    tp_sum = ((y_true == 1) and (y_pred == 1)).sum()
    fn_sum = ((y_true == 1) and (y_pred == 0)).sum()
    return tp_sum / (tp_sum + fn_sum)


def f_score(y_true, y_pred, beta=1):
    tp_sum = ((y_true == 1) and (y_pred == 1)).sum()
    fp_sum = ((y_true == 0) and (y_pred == 1)).sum()
    fn_sum = ((y_true == 1) and (y_pred == 0)).sum()
    numerator = (1 + beta ** 2) * tp_sum
    denominator = (1 + beta ** 2) * tp_sum + beta ** 2 * fn_sum + fp_sum
    return numerator / denominator


def roc_auc_score(y_true, y_score):
    pass