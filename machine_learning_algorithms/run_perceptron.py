import numpy as np
import matplotlib.pyplot as plt

from models.perceptron import Perceptron


# Data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype='float32')
y = np.array([[0], [0], [0], [1]], dtype='float32')

# Define and train model
model = Perceptron(data=X, labels=y, num_input=2)
model.fit(alpha=0.1, epochs=5000)

# Print results
print('x1\tx2\tlabel\tprediction')
for i in range(X.shape[0]):
    print('{x1}\t{x2}\t{label}\t{prediction}'.format(
        x1=X[i, 0],
        x2=X[i, 1],
        label=y[i, 0],
        prediction=model.predict(X[i, :])[0][0]))

# Plot results
weights = model.w
bias = model.b

x_fit, y_fit = np.linspace(-1, 2, 100), []
for x in x_fit:
    y_fit.append(-(weights[0, 0] * x + bias[0, 0]) / weights[0, 1])

plt.scatter(X[:, 0], X[:, 1], color='blue', label='Samples')
plt.plot(x_fit, y_fit, color='red', label='Decision Boundary')
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.legend()
plt.show()
