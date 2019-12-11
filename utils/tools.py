import numpy as np


def add_constant(X):
    return np.insert(X, 0, 1, axis=1)