import numpy as np


class LinearRegression:
    """Ordinary least squares via closed-form normal equation or gradient descent."""

    def __init__(self, method="closed_form", lr=0.01, n_iters=1000):
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
