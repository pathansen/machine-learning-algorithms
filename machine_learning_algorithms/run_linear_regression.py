import random as random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from models.linear_regression import LinearRegression

# Use custom styling from file
matplotlib.rc_file('../plotstyle')

# Generate data
random.seed(0)
X = np.array([i for i in range(20)], dtype='float32')
X = np.reshape(X, (20, 1))
X = np.concatenate((np.ones((20, 1), dtype='float32'), X), axis=1)

y = np.array([(i + random.uniform(-2, 2)) for i in range(20)], dtype='float32')
y = np.reshape(y, (20, 1))

# Fit model to data
model = LinearRegression(data=X, labels=y)
weights = model.fit()

# Generate line of best fit
x_bf = np.linspace(0, 20, dtype='float32')
y_bf = np.array([(weights[0][0] + x * weights[1][0]) for x in x_bf],
                dtype='float32')

plt.scatter(X[:, 1], y, color='b', s=50, label='Samples')
plt.plot(x_bf, y_bf, color='r', label='Fitted Model')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Linear Regression')
plt.legend()
plt.show()
