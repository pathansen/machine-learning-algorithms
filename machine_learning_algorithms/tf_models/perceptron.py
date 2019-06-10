import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tfe.enable_eager_execution()


class Perceptron(object):

    def __init__(self, data, labels, num_input):
        """
        Constructor for Perceptron class.

        :param data: Design matrix of shape `(num_samples, num_features)` \\
        :param labels: Data labels of shape `(num_samples, 1)` \\
        :param num_input: Number of input neurons
        """
        self.data = data
        self.labels = labels
        self.num_input = num_input

        # Weights and biases
        self.w = tfe.Variable(np.random.rand(1,
                              self.num_input).astype(np.float32))
        self.b = tfe.Variable(np.random.rand(1, 1).astype(np.float32))

    def fit(self, epochs=1000, alpha=0.001, beta=0, verbose=True):
        """
        Find optimal weights to fit to samples.

        :param epochs: Number of epochs to train model for \\
        :param alpha: Learning rate hyperparameter \\
        :param beta: Regularization hyperparameter \\
        :param verbose: Flag for verbose logging \\
        :return: Optimal weights
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

    def predict(self, x):
        x = x.reshape(self.data.shape[1], 1)
        return tf.math.sigmoid(tf.matmul(self.w, x) + self.b)

    def _model(self):
        """
        TensorFlow model for Perceptron.

        :return: Output of sigmoid function for passed input
        """
        X, w, b = self.data, self.w, self.b
        return tf.math.sigmoid(tf.matmul(X, tf.transpose(w)) + b)

    def _loss(self, beta):
        """
        Loss function for Perceptron model.

        :param beta: Regularization hyperparameter \\
        :return: Loss function
        """
        X, y = self.data, self.labels
        loss = tf.reduce_sum(tf.abs(self._model() - y))
        return loss
