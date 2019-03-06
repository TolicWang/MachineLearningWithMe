# @Time    : 2019/3/6 14:36
# @Email  : wangchengo@126.com
# @File   : plot002.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth

centers = [[1, 1], [5, 1]]  # 指定簇中心
x, y = make_blobs(n_samples=400, centers=centers, cluster_std=1.2, random_state=np.random.seed(100))
size = 35
for i in range(400):
    color = 'orange' if y[i] == 0 else 'red'
    mark = 'o' if y[i] == 0 else 's'  # 形状
    plt.scatter(x[i, 0], x[i, 1], c=color, marker=mark, alpha=0.83, cmap='hsv', s=size, )  # alpha 控制透明度

plt.scatter(centers[0][0], centers[0][1], c='black', alpha=0.83, cmap='hsv', s=70)  # 簇中心点
plt.scatter(centers[1][0], centers[1][1], c='black', alpha=0.83, cmap='hsv', s=70)  # 簇中心点

plt.annotate(r'$V_1$', xy=(5, 1), fontsize=13, color='black',
             xytext=(7, 1.5), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
plt.annotate(r'$V_2$', xy=(1, 1), fontsize=13, color='black',
             xytext=(-0.4, 2.4), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
plt.annotate(r'$\;$', xy=(1, 1), fontsize=13, color='red',
             xytext=(3.8, 0.75), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='blue'))
plt.annotate(r'$\;$', xy=(5, 1), fontsize=13, color='red',
             xytext=(3.7, 0.74), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='green'))
plt.annotate(r'$d$', xy=(1, 1), fontsize=22, color='blue',
             xytext=(2.1, 0.38))
plt.annotate(r'$D$', xy=(1, 1), fontsize=20, color='green',
             xytext=(4.3, 0.3))
plt.show()
