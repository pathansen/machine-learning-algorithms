import os
import random as random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tfe.enable_eager_execution()


class LogisticRegression(object):

    def __init__(self, data, labels):
        """
        Constructor for Logistic Regression class.

        :param data: Design matrix of shape `(num_samples, num_features)` \\
        :param labels: Data labels of shape `(num_samples, 1)` \\
        """
        self.data = data
        self.labels = labels
        self.w = tfe.Variable(np.random.rand(
                             data.shape[1],
                             labels.shape[1]).astype(np.float32))

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

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
        gradient = tfe.implicit_gradients(self._loss)

        for epoch in range(epochs):
            optimizer.apply_gradients(gradient(beta))

            if verbose:
                once_every = epochs // 10
                if (epoch + 1) % once_every == 0:
                    loss = self._loss(beta)
                    print('Epoch: %i --> loss = %0.3f' % (epoch + 1, loss))

        return self.w.numpy()

    def predict(self, sample):
        """
        Predict value for a given input sample.

        :param sample: Input sample of shape `(num_features, 1)` \\
        :return: Predicted value
        """
        x = sample
        return tf.math.sigmoid(np.matmul(x, self.w))

    def _model(self):
        """
        TensorFlow model for Logistic Regression.

        :return: Output of sigmoid function for passed input
        """
        X, w = self.data, self.w
        return tf.math.sigmoid(tf.matmul(X, w))

    def _loss(self, beta):
        """
        Loss function for Logistic Regression model.

        :param beta: Regularization hyperparameter \\
        :return: Loss function
        """
        X, y, w = self.data, self.labels, self.w
        loss = tf.reduce_sum(
            -y * tf.log(self._model()) - (1 - y) *
            tf.log(1 - self._model()) +
            beta * tf.nn.l2_loss(w))
        return loss
