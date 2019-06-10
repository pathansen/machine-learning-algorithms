import os
import random as random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tfe.enable_eager_execution()


class LinearRegression(object):

    def __init__(self, data, labels):
        """
        Constructor for Linear Regression class.

        :param data: Design matrix of shape `(num_samples, num_features)` \\
        :param labels: Data labels of shape `(num_samples, 1)`
        """
        self.data = data
        self.labels = labels
        self.w = None

    def fit(self, beta=0):
        """
        Find optimal weights to fit to samples.

        :param beta: Regularization parameter \\
        :return: Optimal weights
        """
        X, y = self.data, self.labels

        if beta == 0:
            R = tf.matmul(X.T, X)
            P = tf.matmul(X.T, y)
        else:
            R = tf.matmul(X.T, X) + beta * tf.identity(X.shape[1])
            P = tf.matmul(X.T, y)
        self.w = tf.matmul(tf.linalg.inv(R), P)
        return self.w

    def predict(self, sample):
        """
        Predict value for a given input sample.

        :param sample: Input sample of shape `(num_features, 1)` \\
        :return: Predicted value
        """
        x = sample
        return tf.matmul(x, self.w)


if __name__ == '__main__':
    print('working')
