import numpy as np


class NeuralNetwork(object):

    def __init__(self, data, labels, num_input, num_hidden, hidden_nodes, num_output):
        """
        Constructor for Neural Network class.

        :param data: Design matrix of shape `(num_samples, num_features)` \\
        :param labels: Data labels of shape `(num_samples, 1)` \\
        :param num_input: Number of input neurons \\
        :param num_hidden: Number of layers \\
        :param hidden_nodes: Number of neurons in each hidden layer \\
        :param num_output: Number of output neurons
        """
        self.data = data
        self.labels = labels
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.hidden_nodes = hidden_nodes
        self.num_output = num_output

        # Need to define weights and biases here...
        self.w = [None for i in range(self.num_hidden + 1)]
        self.b = [None for i in range(self.num_hidden + 1)]

        for i in range(self.num_hidden + 1):
            if i == 0:  # input layer
                self.w[i] = np.random.rand(self.hidden_nodes[i],
                                           self.num_input)
                self.b[i] = np.random.rand(self.hidden_nodes[i], 1)
            elif i == self.num_hidden:  # output layer
                self.w[i] = np.random.rand(self.num_output,
                                           self.hidden_nodes[i - 1])
                self.b[i] = np.random.rand(self.num_output, 1)
            else:  # all other layers
                self.w[i] = np.random.rand(self.hidden_nodes[i],
                                           self.hidden_nodes[i - 1])
                self.b[i] = np.random.rand(self.hidden_nodes[i], 1)

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

        layer = [None for i in range(self.num_hidden + 1)]
        layer_error = [None for i in range(self.num_hidden + 1)]

        training_loss = []
        for epoch in range(epochs):
            loss = 0
            for n in range(X.shape[0]):

                # Feedforward
                x = X[n, :].reshape(X.shape[1], 1)
                for i in range(self.num_hidden + 1):
                    if i == 0:  # input layer
                        layer[i] = self._relu_m(np.matmul(
                            self.w[i], x) + self.b[i])
                    elif i == self.num_hidden:  # output layer
                        layer[i] = self._sigmoid_m(np.matmul(
                            self.w[i], layer[i - 1]) + self.b[i])
                    else:  # all other layer
                        layer[i] = self._relu_m(np.matmul(
                            self.w[i], layer[i - 1]) + self.b[i])

                # Back propagation
                error = layer[-1] - y[n, 0]
                loss += abs(sum(error[0]))
                for i in range(self.num_hidden, -1, -1):
                    if i == self.num_hidden:  # output layer
                        layer_error[i] = self._backpropagation_output(
                            alpha=alpha,
                            index=i,
                            error=error,
                            layer_input=layer[i - 1],
                            layer_output=layer[i])
                    elif i == 0:  # first hidden layer
                        layer_error[i] = self._backpropagation_hidden(
                            alpha=alpha,
                            index=i,
                            forward_error=layer_error[i + 1],
                            layer_input=x,
                            layer_output=layer[i])
                    else:  # all other layers
                        layer_error[i] = self._backpropagation_hidden(
                            alpha=alpha,
                            index=i,
                            forward_error=layer_error[i + 1],
                            layer_input=layer[i - 1],
                            layer_output=layer[i])

            loss /= X.shape[0]
            training_loss.append(loss)

            if verbose:
                once_every = epochs // 10
                if (epoch + 1) % once_every == 0:
                    print('Epoch: %i --> loss = %0.3f' % (epoch + 1, loss))

        return training_loss

    def predict(self, x):
        x = x.reshape(self.data.shape[1], 1)
        layer = [None for i in range(self.num_hidden + 1)]
        for i in range(self.num_hidden + 1):
            if i == 0:  # input layer
                layer[i] = self._relu_m(np.matmul(
                    self.w[i], x) + self.b[i])
            elif i == self.num_hidden:  # output layer
                layer[i] = self._sigmoid_m(np.matmul(
                    self.w[i], layer[i - 1]) + self.b[i])
            else:  # all other layer
                layer[i] = self._relu_m(np.matmul(
                    self.w[i], layer[i - 1]) + self.b[i])
        return layer[-1]

    def _backpropagation_output(self, alpha, index, error, layer_input, layer_output):
        """
        Backpropagation for output layer of neural network.

        :param alpha: Learning rate \\
        :param weights: Output layer weights \\
        :param biases: Output layer biases \\
        :param error: Error on output layer \\
        :param layer_input: Input to output layer \\
        :param layer_output: Output from output layer \\
        :return: Error from layer
        """
        for j in range(self.w[index].shape[0]):
            for k in range(self.w[index].shape[1]):
                self.w[index][j, k] -= alpha * error[j, 0] \
                    * layer_input[k, 0] \
                    * self._sigmoid(layer_output[j, 0], True)
            self.b[index][j, 0] -= alpha * error[j, 0] \
                * self._sigmoid(layer_output[j, 0], True)

        return error

    def _backpropagation_hidden(self, alpha, index, forward_error, layer_input, layer_output):
        """
        Backpropagation for output layer of neural network.

        :param alpha: Learning rate \\
        :param weights: Output layer weights \\
        :param biases: Output layer biases \\
        :param forward_error: Error from forward layer(s) \\
        :param forward_weights: Weights from forward layer \\
        :param layer_input: Input to output layer \\
        :param layer_output: Output from output layer \\
        :return: Error from layer
        """
        # Computer layer error
        layer_error = np.zeros((self.w[index + 1].shape[1], 1), dtype='float32')
        for j in range(self.w[index + 1].shape[0]):
            for k in range(self.w[index + 1].shape[1]):
                layer_error[k, 0] += self.w[index + 1][j, k] \
                    * forward_error[j, 0] * self._relu(layer_output[k, 0], True)

        # Update weights and biases
        for j in range(self.w[index].shape[0]):
            for k in range(self.w[index].shape[1]):
                self.w[index][j, k] -= alpha * layer_input[k, 0] \
                    * layer_error[j, 0]
            self.b[index][j, 0] -= alpha * layer_error[j, 0]

        return layer_error

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

    def _relu(self, x, deriv=False):
        """
        Rectified Linear Unit (ReLU) activation function.

        :param x: Input to ReLU function \\
        :return: Output of ReLU function for passed input
        """
        if not deriv:
            if x < 0:
                return 0.01 * x
            else:
                return x
        else:
            if x < 0:
                return 0.01
            else:
                return 1.0

    def _relu_m(self, X):
        result = np.zeros((X.shape[0], X.shape[1]), dtype='float32')
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] < 0:
                    result[i, j] = 0.01 * X[i, j]
                else:
                    result[i, j] = X[i, j]
        return result
