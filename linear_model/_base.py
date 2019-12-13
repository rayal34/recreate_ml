import numpy as np
from utils.tools import add_constant
from utils.math import sigmoid


class LinearRegression():
    '''
    Linear regression model using SGD
    '''

    def __init__(self, n_iterations=1000, learning_rate=1e-3):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.coefficients = None

    def fit(self, X, y):
        X = add_constant(X)

        self.coefficients = np.random.random(X.shape[1]) * .001

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.coefficients)
            grad = -np.dot(X.T, y - y_pred)

            self.coefficients -= self.learning_rate * grad

        return self

    def predict(self, X):
        X = add_constant(X)

        return np.dot(X, self.coefficients)


class LogisticRegression:

    def __init__(self, n_iterations=1000, learning_rate=1e-3):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = add_constant(X)

        self.coefficients = np.random.random(X.shape[1]) * .001

        for _ in range(self.n_iterations):
            z = np.dot(X, self.coefficients)
            y_pred = sigmoid(z)

            grad = -np.dot(X.T, y_pred - y)

            self.coefficients -= self.learning_rate * grad

        return self

    def predict_proba(self, X):
        X = add_constant(X)
        z = np.dot(X, self.coefficients)
        y_pred = sigmoid(z)
        return y_pred

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas > .5) * 1

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).mean()