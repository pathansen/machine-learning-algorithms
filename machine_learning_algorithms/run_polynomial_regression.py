import random as random
import numpy as np
import matplotlib.pyplot as plt

from models.polynomial_regression import PolynomialRegression


# Generate data
random.seed(0)
r = 8

X = np.linspace(0, 10, 20, dtype='float32')
y = np.array([(0.01*x**3 + 0.1*x**2 + x + random.uniform(-r, r)) for x in X],
             dtype='float32')

X = np.reshape(X, (20, 1))
X = np.concatenate((np.ones((20, 1), dtype='float32'), X), axis=1)
y = np.reshape(y, (20, 1))

# Fit model to data
order = 6
model = PolynomialRegression(data=X, labels=y)
weights = model.fit(order=order, beta=0.1)

# Generate line of best fit
x_bf = np.linspace(0, 10, dtype='float32')

y_bf = np.zeros(50, dtype='float32')
for i in range(50):
    for j in range(order + 1):
        y_bf[i] += weights[j][0] * x_bf[i]**j


plt.scatter(X[:, 1], y, color='b', label='Samples')
plt.plot(x_bf, y_bf, color='r', label='Fitted Model')
plt.xlim(-2, 12)
plt.ylim(-5, 45)
plt.legend()
plt.show()
