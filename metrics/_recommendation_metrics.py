import numpy as np


def _calc_dcg(y_true, y_score, k):
    ranked_positions = np.argsort(y_score)
    print(y_true)
    print(y_score, ranked_positions)
    scores = np.zeros(y_true.shape[0])
    for i, (ranked_position, relevance) in enumerate(zip(ranked_positions, y_true)):
        # need [::-1] to get descending sort
        ranked_relevance = relevance[ranked_position[::-1]]
        score = sum((r / np.log2(i + 1) for i, r in enumerate(ranked_relevance[:k], 1)))
        scores[i] = score

    return scores


def dcg_score(y_true, y_score, k=None):
    if k is None:
        k = len(y_true[0])

    scores = _calc_dcg(y_true, y_score, k=k)

    return scores.mean()


def ndcg_score(y_true, y_score, k=None):
    dcg_scores = _calc_dcg(y_true, y_score, k=k)
    ndcg_scores = _calc_dcg(y_true, y_true, k)
    return (dcg_scores / ndcg_scores).mean()


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