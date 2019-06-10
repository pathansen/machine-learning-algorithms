import random as random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tf_models.neural_network import NeuralNetwork

# Use custom styling from file
matplotlib.rc_file('../plotstyle')

# Data for XOR gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float32')
y = np.array([[0], [1], [1], [0]], dtype='float32')

# Define and train model
model = NeuralNetwork(X, y, 2, 1, [10], 1)
loss = model.fit(alpha=0.1, epochs=1000, verbose=True)

# Print results
print('\nx1\tx2\tlabel\tprediction')
for i in range(X.shape[0]):
    print('{x1}\t{x2}\t{label}\t{prediction}'.format(
        x1=X[i, 0],
        x2=X[i, 1],
        label=y[i, 0],
        prediction=model.predict(X[i, :])[0][0]))

plt.plot(loss)
plt.xlabel('Training Epoch')
plt.ylabel('Training Loss')
plt.title('Fully Connected Neural Network')
plt.show()
