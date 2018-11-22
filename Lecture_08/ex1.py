# @Time    : 2018/11/22 8:17
# @Email  : wangchengo@126.com
# @File   : as.py
# package info:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
import numpy as np
from tools.accFscore import get_acc_fscore


def load_data():
    from sklearn.preprocessing import StandardScaler
    data = load_digits()
    x = data.data
    y = data.target
    ss = StandardScaler()
    x = ss.fit_transform(x)
    shuffle_index = np.random.permutation(x.shape[0])
    return x[shuffle_index], y[shuffle_index]


def visilize():
    from tools.visiualImage import visiualization
    digits = load_digits()
    visiualization(digits.images, label=digits.target, label_name=digits.target_names)


def Kmeans_model():
    x_train, y_train, = load_data()
    model = KMeans(n_clusters=10)
    model.fit(x_train)
    y_label = model.labels_
    print("------------kmeans聚类结果------------")
    print("轮廓系数", silhouette_score(x_train, y_label))
    print("召回率：%f,准确率: %f"%(get_acc_fscore(y_train, y_label)))

def DBSCAN_model():
    x_train, y_train, = load_data()
    model = DBSCAN(eps=3, min_samples=5)
    model.fit(x_train)
    y_label = model.labels_
    print("------------DBSCAN聚类结果------------")
    print("轮廓系数", silhouette_score(x_train, y_label))
    print("召回率：%f,准确率: %f" % (get_acc_fscore(y_train, y_label)))


if __name__ == "__main__":
    visilize()
    Kmeans_model()
    DBSCAN_model()

