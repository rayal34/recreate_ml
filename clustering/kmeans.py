import numpy as np
from utils.distances import euclidean


import numpy as np


class KMeans:

    def __init__(self, k=5, max_iterations=100, metric='euclidean', random_state=None):
        self.k = k
        self.metric = metric
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Randomly initiate cluster centers
        self.cluster_centers_ = np.random.permutation(X)[:self.k]

        for i in range(self.max_iterations):
            self.labels_ = np.array([self._nearest(self.cluster_centers_, x)
                                     for x in X])

            # Loop through the unique labels, subset the data by the specific
            # label and calculate a new center on the subset data
            new_centers = [X[self.labels_ == label].mean(axis=0)
                           for label in np.unique(self.labels_)]

            # Stop the iteration if the cluster centers don't change
            if set(map(tuple, new_centers)) == set(map(tuple, self.cluster_centers_)):
                print(f'Early convergence at iteration {i}')
                break

            self.cluster_centers_ = np.array(new_centers)

        return self

    def predict(self, X):
        return [self._nearest(self.cluster_centers_, x) for x in X]

    def _nearest(self, clusters, sample):
        distances = euclidean(clusters, sample, axis=1)
        return np.argmin(distances)