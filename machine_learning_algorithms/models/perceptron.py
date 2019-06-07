import numpy as np


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
        self.w = np.random.rand(1, self.num_input)
        self.b = np.random.rand(1, 1)

    def fit(self, epochs=1000, alpha=0.001, beta=0, verbose=False):
        """
        Find optimal weights to fit to samples.

        :param epochs: Number of epochs to train model for \\
        :param alpha: Learning rate hyperparameter \\
        :param beta: Regularization hyperparameter \\
        :param verbose: Flag for verbose logging \\
        :return: Optimal weights
        """
        X, y = self.data, self.labels

        for epoch in range(epochs):
            for n in range(X.shape[0]):

                # Feedforward
                x = X[n, :].reshape(X.shape[1], 1)
                prediction = self._sigmoid_m(np.matmul(self.w, x) + self.b)

                # Back propagation
                error = prediction - y[n, 0]
                self._backpropagation(
                    alpha=alpha,
                    error=error,
                    layer_input=x,
                    layer_output=prediction)

    def predict(self, x):
        x = x.reshape(self.data.shape[1], 1)
        return self._sigmoid_m(np.matmul(self.w, x) + self.b)

    def _backpropagation(self, alpha, error, layer_input, layer_output):
        """
        Backpropagation for output layer of neural network.

        :param alpha: Learning rate \\
        :param error: Error on output layer \\
        :param layer_input: Input to output layer \\
        :param layer_output: Output from output layer \\
        :return: Error from layer
        """
        for j in range(self.w.shape[0]):
            for k in range(self.w.shape[1]):
                self.w[j, k] = self.w[j, k] - alpha * error[j, 0] \
                    * layer_input[k, 0] * self._sigmoid(layer_output[j, 0],
                                                        True)
            self.b[j, 0] = self.b[j, 0] - alpha * error[j, 0] \
                * self._sigmoid(layer_output[j, 0], True)

    def _sigmoid(self, x, deriv=False):
        """
        Sigmoid activation function.

        :param x: Input to sigmoid function \\
        :param deriv: Flag to return Sigmoid of derivative of Sigmoid \\
        :return: Output of sigmoid function for passed input
        """
        if deriv is False:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return self._sigmoid(x, False) * (1 - self._sigmoid(x, False))

    def _sigmoid_m(self, X):
        """
        Sigmoid activation function for matrices (used for entire layers).

        :param x: Input to sigmoid function \\
        :return: Output of sigmoid function for passed input
        """
        result = np.zeros((X.shape[0], X.shape[1]), dtype='float32')
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                result[i, j] = 1.0 / (1.0 + np.exp(-X[i, j]))
        return result
