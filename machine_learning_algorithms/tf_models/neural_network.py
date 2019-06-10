import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tfe.enable_eager_execution()


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

        # Weights and biases
        self.w = [None for i in range(self.num_hidden + 1)]
        self.b = [None for i in range(self.num_hidden + 1)]

        for i in range(self.num_hidden + 1):
            if i == 0:  # input layer
                self.w[i] = tfe.Variable(np.random.rand(self.hidden_nodes[i],
                                         self.num_input).astype(np.float32))
                self.b[i] = tfe.Variable(np.random.rand(self.hidden_nodes[i],
                                         1).astype(np.float32))
            elif i == self.num_hidden:  # output layer
                self.w[i] = tfe.Variable(np.random.rand(self.num_output,
                                         self.hidden_nodes[i - 1]).astype(np.float32))
                self.b[i] = tfe.Variable(np.random.rand(self.num_output,
                                         1).astype(np.float32))
            else:  # all other layers
                self.w[i] = tfe.Variable(np.random.rand(self.hidden_nodes[i],
                                         self.hidden_nodes[i - 1]).astype(np.float32))
                self.b[i] = tfe.Variable(np.random.rand(self.hidden_nodes[i],
                                         1).astype(np.float32))

        self.theta = self.w + self.b

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

        optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
        gradient = tfe.implicit_gradients(self._loss)

        training_loss = []
        for epoch in range(epochs):
            optimizer.apply_gradients(gradient(beta))

            if verbose:
                once_every = epochs // 10
                if (epoch + 1) % once_every == 0:
                    loss = np.mean(self._loss(beta).numpy())
                    training_loss.append(loss)
                    print('Epoch: %i --> loss = %0.3f' % (epoch + 1, loss))

        return training_loss

    def predict(self, x):
        x = x.reshape(self.data.shape[1], 1)
        layer = [None for i in range(self.num_hidden + 1)]
        for i in range(self.num_hidden + 1):
            if i == 0:  # input layer
                layer[i] = tf.nn.relu(tf.matmul(self.w[i], x) + self.b[i])
            elif i == self.num_hidden:  # output layer
                layer[i] = tf.nn.sigmoid(tf.matmul(self.w[i], layer[i - 1]) +
                                         self.b[i])
            else:  # all other layer
                layer[i] = tf.nn.relu(tf.matmul(self.w[i], layer[i - 1]) +
                                      self.b[i])
        return layer[-1]

    def _model(self):
        """
        TensorFlow model for Neural Network.

        :return: Output of sigmoid function for passed input
        """
        X, w = self.data, self.w
        layer = [None for i in range(self.num_hidden + 1)]
        for i in range(self.num_hidden + 1):
            if i == 0:  # input layer
                layer[i] = tf.nn.relu(tf.matmul(self.w[i], X.T) + self.b[i])
            elif i == self.num_hidden:  # output layer
                layer[i] = tf.matmul(self.w[i], layer[i - 1]) + self.b[i]
            else:  # all other layer
                layer[i] = tf.nn.relu(tf.matmul(self.w[i], layer[i - 1]) +
                                      self.b[i])
        return layer[-1]

    def _loss(self, beta):
        """
        Loss function for Neural Network model.

        :param beta: Regularization hyperparameter \\
        :return: Loss function
        """
        X, y, w = self.data, self.labels, self.w
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self._model(),
                labels=y.T))
        l2_loss = tf.add_n([tf.nn.l2_loss(t) for t in self.theta])
        return loss + beta * l2_loss
