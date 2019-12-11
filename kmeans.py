import numpy as np
from utils.distances import euclidean


class KMeans:

    def __init__(self, k=5, iterations=100, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.iterations = iterations

    def fit(self, X):
        self.X = X

