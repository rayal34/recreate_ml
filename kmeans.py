import numpy as np
from utils import euclidean


class KMeans:

    def __init__(self, k=5, iterations=100, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.iterations = iterations

    def fit(self, X):
        self.X = X
        centroids = np.random.random(self.X.shape)
        for i in range(iterations):



    def predict(self, X):
        distances = np.zeros((X.shape[0], self.X.shape[0]))
        predictions = np.zeros(X.shape[0])
        distance_func = self._get_distance_measure()
        for i, pt in enumerate(X):
            for j, train_pt in enumerate(self.X):
                distances[i, j] = distance_func(pt, train_pt)

            nearest = np.argsort(distances[i])[:self.n_neighbors]
            nearest_classes = self.y[nearest]
            class_cnts = np.bincount(nearest_classes)
            predictions[i] = class_cnts.argmax()

        return predictions