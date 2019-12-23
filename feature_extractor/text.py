import numpy as np
from preprocessing import normalize


class Tfidf:

    def __init__(self, norm='l2'):
        self.idf_ = None
        self.norm = norm

    def _calc_idf(self, X):
        df = np.sum(X > 0, axis=0)
        n = X.shape[0]
        return 1 + np.log((1 + n) / (1 + df))

    def fit(self, X):

        self.idf_ = self._calc_idf(X)

    def transform(self, X):

        tfidf = np.multiply(X, self.idf_)

        if self.norm == 'l2':
            tfidf = tfidf / normalize(tfidf, p=2, axis=1).reshape((-1, 1))
        elif self.norm == 'l1':
            tfidf = tfidf / normalize(tfidf, p=1, axis=1).reshape((-1, 1))

        return tfidf