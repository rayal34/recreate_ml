import numpy as np


def normalize(X, p=2, axis=0):
    return (np.sum(np.abs(X) ** p)) ** (1 / p)