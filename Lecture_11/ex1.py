# @Time    : 2018/12/13 10:26
# @Email  : wangchengo@126.com
# @File   : ex1.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

import numpy as np
import pandas as pd
import scipy.io as load
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def costFandGradient(X, y_label, W1, b1, W2, b2, lambd):
    # ============    forward propogation
    m, n = np.shape(X)  # m:samples, n: dimensions
    a1 = X  # 5000 by 400
    z2 = np.dot(a1, W1) + b1  # 5000 by 400 dot 400 by 25 + 25 by 1= 5000 by 25
    a2 = sigmoid(z2)  # 5000 by 25
    z3 = np.dot(a2, W2) + b2  # 5000 by 25 dot 25 by 10 + 10 by 1= 5000 by 10
    a3 = sigmoid(z3)  # 5000 by 10
    cost = (1 / m) * np.sum((a3 - y_label) ** 2) + (lambd / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    # ===========   back propogation
    delta3 = -(y_label - a3) * sigmoidGradient(z3)  # 5000 by 10
    df_w2 = np.dot(a2.T, delta3)  # 25 by 5000 dot 5000 by 10 = 25 by 10
    df_w2 = (1 / m) * df_w2 + lambd * W2

    delta2 = np.dot(delta3, W2.T) * sigmoidGradient(z2)  # =5000 by 10 dot 10 by 25 = 5000 by 25
    df_w1 = np.dot(a1.T, delta2)  # 400 by 5000 dot 5000 by 25 = 400 by 25
    df_w1 = (1 / m) * df_w1 + lambd * W1

    df_b1 = (1 / m) * np.sum(delta2, axis=0)
    df_b2 = (1 / m) * np.sum(delta3, axis=0)
    return cost, df_w1, df_w2, df_b1, df_b2


def gradientDescent(learn_rate, W1, b1, W2, b2, df_w1, df_w2, df_b1, df_b2):
    W1 = W1 - learn_rate * df_w1  # 400,25
    W2 = W2 - learn_rate * df_w2  # 25,10
    b1 = b1 - learn_rate * df_b1  # 25 by 1
    b2 = b2 - learn_rate * df_b2  # 10 by 1
    return W1, b1, W2, b2


def load_data():
    data = load.loadmat('./data/ex4data1.mat')
    X = data['X']  # 5000 by 400  samples by dimensions
    y = data['y'].reshape(5000)
    eye = np.eye(10)
    y_label = eye[y - 1, :]  # 10 by 5000
    ss = StandardScaler()
    X = ss.fit_transform(X)
    return X, y, y_label


def train():
    X, y, y_label = load_data()
    input_layer_size = 400
    hidden_layer_size = 25
    output_layer_size = 10
    epsilong_init = 0.15
    W1 = np.random.rand(input_layer_size, hidden_layer_size) * 2 * epsilong_init - epsilong_init
    W2 = np.random.rand(hidden_layer_size, output_layer_size) * 2 * epsilong_init - epsilong_init
    b1 = np.random.rand(hidden_layer_size) * 2 * epsilong_init - epsilong_init
    b2 = np.random.rand(output_layer_size) * 2 * epsilong_init - epsilong_init

    lambd = 0.0
    iteration = 5000
    cost = []
    learn_rate = 0.7
    for i in range(iteration):
        c, df_w1, df_w2, df_b1, df_b2 = costFandGradient(X, y_label, W1, b1, W2, b2, lambd)
        cost.append(round(c, 4))
        W1, b1, W2, b2 = gradientDescent(learn_rate, W1, b1, W2, b2, df_w1, df_w2, df_b1, df_b2)
        print('loss--------------', c)
    p = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    temp = open('./data/para.pkl', 'wb')
    pickle.dump(p, temp)

    x = np.arange(1, iteration + 1)
    plt.plot(x, cost)
    plt.show()


def prediction():
    X, y, y_label = load_data()
    p = open('./data/para.pkl', 'rb')
    data = pickle.load(p)
    W1 = data['W1']
    W2 = data['W2']
    b1 = data['b1']
    b2 = data['b2']
    a1 = X  # 5000 by 400
    z2 = np.dot(a1, W1) + b1  # 5000 by 400 dot 400 by 25 + 25 by 1= 5000 by 25
    a2 = sigmoid(z2)  # 5000 by 25
    z3 = np.dot(a2, W2) + b2  # 5000 by 25 dot 25 by 10 + 10 by 1= 5000 by 10
    a3 = sigmoid(z3)  # 5000 by 10
    y_pre = np.zeros(a3.shape[0], dtype=int)
    for i in range(a3.shape[0]):
        col = a3[i,:]
        index = np.where(col == np.max(col))[0][0] + 1
        y_pre[i] = index
    print(accuracy_score(y, y_pre))


if __name__ == '__main__':
    # load_data()
    train()
    prediction()
