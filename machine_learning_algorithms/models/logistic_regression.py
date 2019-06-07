import random as random
import numpy as np


class LogisticRegression(object):

    def __init__(self, data, labels):
        """
        Constructor for Logistic Regression class.

        :param data: Design matrix of shape `(num_samples, num_features)` \\
        :param labels: Data labels of shape `(num_samples, 1)` \\
        """
        self.data = data
        self.labels = labels
        self.w = np.random.rand(data.shape[1], labels.shape[1])

    def fit(self, epochs=1000, alpha=0.001, beta=0, verbose=False):
        """
        Find optimal weights to fit to samples.

        :param epochs: Number of epochs to train model for \\
        :param alpha: Learning rate hyperparameter \\
        :param beta: Regularization hyperparameter \\
        :param verbose: Flag for verbose logging \\
        :return: Optimal weights \\
        """
        X, y = self.data, self.labels

        for epoch in range(epochs):
            for n in range(X.shape[0]):
                mu = self._sigmoid(np.matmul(X[n, :].T, self.w))
                loss = -y[n, 0] * np.log2(mu) - (1 - y[n, 0]) * np.log2(1 - mu)
                delta_w = X[n, :].T * (mu - y[n, 0])
                self.w -= alpha * delta_w.reshape(X.shape[1], y.shape[1]) + \
                    beta * self.w

            if verbose:
                once_every = epochs // 10
                if (epoch + 1) % once_every == 0:
                    print('Epoch: %i --> loss = %0.3f' % (epoch + 1, loss))

        return self.w

    def predict(self, sample):
        """
        Predict value for a given input sample.

        :param sample: Input sample of shape `(num_features, 1)` \\
        :return: Predicted value
        """
        x = sample
        return self._sigmoid(np.matmul(x, self.w))

    def _sigmoid(self, x):
        """
        Sigmoid function used for link function in GLM.

        :param x: Input to sigmoid function \\
        :return: Output of sigmoid function for passed input
        """
        return 1.0 / (1.0 + np.exp(-1 * x))[0]
