import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LogisticRegression import gradDescent,cost_function,accuracy,feature_scalling


def load_data():
    data = pd.read_csv('./data/LogiReg_data.txt', names=['exam1', 'exam2', 'label']).as_matrix()
    X = data[:, :-1]  # 取前两列
    y = data[:, -1:]  # 取最后一列
    shuffle_index = np.random.permutation(X.shape[0])
    X = X[shuffle_index]
    y = y[shuffle_index]
    return X, y


def visualize_data(X, y):
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    plt.scatter(X[positive,0],X[positive,1],s=30,c='b',marker='o',label='Admitted')
    plt.scatter(X[negative,0],X[negative,1],s=30,c='r',marker='o',label='Not Admitted')
    plt.legend()
    plt.show()

def visualize_cost(ite,cost):
    plt.plot(np.linspace(0,ite,ite),cost,linewidth=1)
    plt.title('cost history',color='r')
    plt.xlabel('iterations')
    plt.ylabel('cost J')
    plt.show()


if __name__ == '__main__':
    # Step 1.  Load data
    X, y = load_data()
    # Step 2.  Visualize data
    visualize_data(X, y)
    #
    m, n = X.shape
    X = feature_scalling(X)
    alpha = 0.1
    W = np.random.randn(n, 1)
    b = 0.1
    maxIt = 10000
    W, b, cost_history = gradDescent(X, y, W, b, alpha, maxIt)
    print("******************")
    print(cost_history[:20])
    visualize_cost(maxIt,cost_history)
    print("accuracys is :         " + str(accuracy(X, y, W, b)))
    print("******************")
