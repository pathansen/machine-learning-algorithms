# Machine Learning Algorithms

This repository contains code for various machine learning algorithms implemented in both Python and TensorFlow (using eager execution). The Python implementation demonstrates how the different machine learning algorithms function through both showing the models fit to the data through updating the parameters via gradient descent and how the model can be used to make predictions.

Complimentary, the TensorFlow models show how the same algorithms can be implemented using TensorFlow eager API. In comparison to the Python implementations, the interface to the models are the same but TensorFlow handles updating the parameters of the model so explicit code to update each parameter is not needed.

Both the Python and TensorFlow implementations of the different machine learning models were built with a common interface. The models are implemented as classes with `fit()` and `predict()` functions which will train the model to fit to the passed dataset and the make predictions on individual samples, respectively.

### Workspace Setup
This code was written using Python 3. For development, Python 3.5 was used due to issues with TensorFlow not running on new versions of Python on Windows but newer versions of Python should work if TensorFlow runs in your environment.

First, create a virtual environment:

    $ pip install virtualenv
    $ virtualenv -p python .venv

Activate the virtual enviornment:

Windows:

    $ source .venv/Scripts/activate

macOS or Linux:

    $ source .venv/bin/activate

Install the requirements:

    $ pip install -r requirements.txt

---

## Linear Regression

$ \hat{y} = Xw $

Python model:

    from models.linear_regression import LinearRegression

    model = LinearRegression(data=X, labels=y)
    model.fit()

TensorFlow model:

    from tf_models.linear_regression import LinearRegression

    model = LinearRegression(data=X, labels=y)
    model.fit()

## Logistic Regression
Python model:

    from models.logistic_regression import LogisticRegression

    model = LogisticRegression(data=X, labels=y)
    model.fit()

TensorFlow model:

    from tf_models.logistic_regression import LogisticRegression

    model = LogisticRegression(data=X, labels=y)
    model.fit()

## Poisson Regression
In progress...

## Perceptron
Python model:

    from models.perceptron import Pereceptron

    model = Pereceptron(data=X, labels=y)
    model.fit()

TensorFlow model:

    from tf_models.perceptron import Pereceptron

    model = Pereceptron(data=X, labels=y)
    model.fit()

## Neural Network
Python model:

    from models.neural_network import NeuralNetwork

    model = NeuralNetwork(data=X, labels=y)
    model.fit()

TensorFlow model:

    from tf_models.neural_network import NeuralNetwork

    model = NeuralNetwork(data=X, labels=y)
    model.fit()
