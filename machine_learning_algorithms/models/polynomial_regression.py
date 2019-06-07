import random as random
import numpy as np


class PolynomialRegression(object):

    def __init__(self, data, labels):
        """
        Constructor for Polynomial Regression class.

        :param data: Design matrix of shape `(num_samples, num_features)` \\
        :param labels: Data labels of shape `(num_samples, 1)`
        """
        self.data = data
        self.labels = labels
        self.w = None

    def fit(self, order=2, beta=0):
        """
        Find optimal weights to fit to samples.

        :param beta: Regularization parameter \\
        :return: Optimal weights
        """
        X, y = self.data, self.labels

        for i in range(order - 1):
            X = np.concatenate((X, np.power(X[:, 1], i + 2).reshape(20, 1)),
                               axis=1)

        if beta == 0:
            R = np.matmul(X.T, X)
            P = np.matmul(X.T, y)
        else:
            R = np.matmul(X.T, X) + beta * np.identity(X.shape[1])
            P = np.matmul(X.T, y)
        self.w = np.matmul(np.linalg.inv(R), P)
        return self.w

    def predict(self, sample):
        """
        Predict value for a given input sample.

        :param sample: Input sample of shape `(num_features, 1)` \\
        :return: Predicted value
        """
        x = sample
        return np.matmul(x, self.w)
