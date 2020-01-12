import numpy as np


def manhattan_dist(a, b):
    return np.sum(np.abs(a - b))


def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2), axis=1))