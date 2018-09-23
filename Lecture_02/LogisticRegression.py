import numpy as np
from sklearn.datasets import load_breast_cancer


def feature_scalling(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def load_data(shuffled=False):
    data_cancer = load_breast_cancer()
    x = data_cancer.data
    y = data_cancer.target
    x = feature_scalling(x)
    y = np.reshape(y, (len(y), 1))
    if shuffled:
        shuffled_index = np.random.permutation(y.shape[0])
        x = x[shuffled_index]
        y = y[shuffled_index]
    return x, y


def sigmoid(z):
    gz = 1 / (1 + np.exp(-z))
    return gz


def gradDescent(X, y, W, b, alpha, maxIt):
    cost_history = []
    maxIteration = maxIt
    m, n = X.shape
    for i in range(maxIteration):
        z = np.dot(X, W) + b
        error = sigmoid(z) - y
        W = W - (1 / m) * alpha * np.dot(X.T, error)
        b = b - (1.0 / m) * alpha * np.sum(error)
        cost_history.append(cost_function(X, y, W, b))
    return W, b, cost_history


def accuracy(X, y, W, b):
    m, n = np.shape(X)
    z = np.dot(X, W) + b
    y_hat = sigmoid(z)
    predictioin = np.ones((m, 1), dtype=float)
    for i in range(m):
        if y_hat[i, 0] < 0.5:
            predictioin[i] = 0.0
    return 1 - np.sum(np.abs(y - predictioin)) / m


def cost_function(X, y, W, b):
    m, n = X.shape
    z = np.dot(X, W) + b
    y_hat = sigmoid(z)
    J = (-1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return J

if __name__ == '__main__':
    X, y = load_data()
    m, n = X.shape
    alpha = 0.1
    W = np.random.randn(n, 1)
    b = 0.1
    maxIt = 200
    W, b, cost_history = gradDescent(X, y, W, b, alpha, maxIt)
    print("******************")
    print("W is :             ")
    print(W)
    print("accuracy is :         " + str(accuracy(X, y, W, b)))
    print("******************")
