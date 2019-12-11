import numpy as np


class MultinomialNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # group by class

        classes = np.unique(y)
        class_data = []
        count = []

        for i, c in enumerate(classes):
            class_data.append(X[np.where(y == c)])
            count.append(class_data[i].sum(axis=0) + self.alpha)

        count = np.asarray(count)

        self.class_log_prior_ = np.log(np.bincount(y) / y.shape[0])

        self.feature_log_prob = np.asarray([np.log(cnt / sum(cnt)) for cnt in count])

        return self

    def predict_log_proba(self, X):
        log_proba = np.asarray([(self.feature_log_prob * x).sum(axis=1) + self.class_log_prior_
                                for x in X])
        return log_proba

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return np.argmax(log_proba, axis=0)


class BernoulliNB(object):
    def __init__(self, alpha=1.0, binarizer=False):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y):

        if self.binarize:
            X = (X > 0) * 1

        classes = np.unique(y)
        class_data = []
        count = []

        for i, c in enumerate(classes):
            class_data.append(X[np.where(y == c)])
            count.append(class_data[i].sum(axis=0) + self.alpha)

        count = np.asarray(count)

        self.class_log_prior_ = np.log(np.bincount(y) / y.shape[0])

        self.feature_log_prob = np.asarray([np.log(cnt / sum(cnt)) for cnt in count])

        return self

    def predict_log_proba(self, X):
        log_proba = np.asarray([(self.feature_log_prob * x).sum(axis=1) + self.class_log_prior_ for x in X])
        return log_proba

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return np.argmax(log_proba, axis=0)

