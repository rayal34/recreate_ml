import numpy as np


def manhattan_dist(a, b, axis=None):
    return np.sum(np.abs(a - b), axis=axis)


def euclidean_dist(a, b, axis=None):
    return np.sqrt(np.sum(np.power(a - b, 2), axis=axis))