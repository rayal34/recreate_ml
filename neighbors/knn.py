import numpy as np
from utils import distances


class KNNClassifier:

    def __init__(self, k, distance='euclidean'):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def _find_neighbors(self, x):

        distances = np.zeros(self.X.shape[0])
        for i, train_point in enumerate(self.X):

            distances[i] = self._measure_distance(x, train_point)

        neighbor_indices = np.argsort(distances)[:self.k]
        neighbors = self.y[neighbor_indices]
        class_cnts = np.bincount(neighbors)

        return class_cnts

    def predict_proba(self, X):
        probas = []
        for x in X:
            probas.append(self._find_neighbors(x) / self.k)

        return probas

    def predict(self, X):
        predictions = np.zeros(X.shape[0])

        probas = self.predict_proba(X)

        for i, proba in enumerate(probas):
            predictions[i] = np.argmax(proba)

        return predictions

    def _measure_distance(self, a, b):
        if self.distance == 'euclidean':
            return distances.euclidean_dist(a, b)
        elif self.distance == 'manhattan':
            return distances.euclidean_dist(a, b)

    def score(self, X, y):
        y_preds = self.predict(X)
        return (y == y_preds).mean()