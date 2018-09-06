from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt


def feature_scalling(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def load_data(shuffled=False):
    data = load_boston()
    # print(data.DESCR)# 数据集描述
    X = data.data
    y = data.target
    X = feature_scalling(X)
    y = np.reshape(y, (len(y), 1))
    if shuffled:
        shuffle_index = np.random.permutation(y.shape[0])
        X = X[shuffle_index]
        y = y[shuffle_index]  # 打乱数据
    return X, y


def costJ(X, y, w, b):
    m, n = X.shape
    J = 0.5 * (1 / m) * np.sum((y - np.dot(X, w) - b) ** 2)
    return J


X, y = load_data()
m, n = X.shape  # 506,13
w = np.random.randn(13, 1)
b = 0.1
alpha = 0.01
cost_history = []
for i in range(5000):
    y_hat = np.dot(X, w) + b
    grad_w = -(1 / m) * np.dot(X.T, (y - y_hat))
    grad_b = -(1 / m) * np.sum(y - y_hat)
    w = w - alpha * grad_w
    b = b - alpha * grad_b
    if i % 100 == 0:
        cost_history.append(costJ(X, y, w, b))

# plt.plot(np.arange(len(cost_history)),cost_history)
# plt.show()
# print(cost_history)

y_pre = np.dot(X, w) + b
numerator = np.sum((y - y_pre) ** 2)
denominator= np.sum((y - y.mean()) ** 2)
print(1 - (numerator / denominator))
