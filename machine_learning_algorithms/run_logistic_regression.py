import random as random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from models.logistic_regression import LogisticRegression

# Use custom styling from file
matplotlib.rc_file('../plotstyle')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1 * x))


# Data from:
# https://en.wikipedia.org/wiki/Logistic_regression#Probability_of_passing_an_exam_versus_hours_of_study
X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
             dtype='float32')
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
             dtype='float32')

X, y = np.reshape(X, (20, 1)), np.reshape(y, (20, 1))
X = np.concatenate((np.ones((20, 1), dtype='float32'), X), axis=1)

# Fit model to data
model = LogisticRegression(data=X, labels=y)
weights = model.fit(alpha=0.1, verbose=True)

# Generate line of best fit
x_bf = np.linspace(0, 6, dtype='float32')
y_bf = np.array([sigmoid(weights[0][0] + x * weights[1][0]) for x in x_bf],
                dtype='float32')

plt.scatter(X[:, 1], y, color='b', s=75, label='Samples')
plt.plot(x_bf, y_bf, color='r', label='Fitted Model')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Logistic Regression')
plt.legend()
plt.show()
