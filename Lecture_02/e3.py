import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LogisticRegression import feature_scalling
from sklearn.linear_model import LogisticRegression

def load_data():
    data = pd.read_csv('./data/LogiReg_data.txt', names=['exam1', 'exam2', 'label']).as_matrix()
    X = data[:, :-1]  # 取前两列
    y = data[:, -1:]  # 取最后一列
    shuffle_index = np.random.permutation(X.shape[0])
    X = X[shuffle_index]
    y = y[shuffle_index]
    return X, y


def visualize_cost(ite,cost):
    plt.plot(np.linspace(0,ite,ite),cost,linewidth=1)
    plt.title('cost history',color='r')
    plt.xlabel('iterations')
    plt.ylabel('cost J')
    plt.show()


if __name__ == '__main__':
    X, y = load_data()
    X = feature_scalling(X)
    lr = LogisticRegression()
    lr.fit(X,y)
    print("******************")
    print("accuracys is :" ,lr.score(X,y))
    print("W:{},b:{}".format(lr.coef_,lr.intercept_))
    print("******************")