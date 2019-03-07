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
from sklearn.datasets.samples_generator import make_blobs, make_circles

centers = [[0, 1], [5, 1], [2.5, 4.5], [8, 8]]  # 指定簇中心

x, y = make_blobs(n_samples=1500, centers=centers, cluster_std=1.0, random_state=np.random.seed(100))

global_center = np.mean(x, axis=0)
size = 35

k = np.unique(y)

markers = ['o', 's', 'd', 'h', '*', '>', '<']
for i in k:
    index = np.where(y == i)[0]
    plt.scatter(x[index, 0], x[index, 1], marker=markers[i], alpha=0.83, s=size, )  # alpha 控制透明度

for center in centers:
    plt.scatter(center[0], center[1], c='black', alpha=0.83, cmap='hsv', s=70)  # 簇中心点

plt.scatter(global_center[0], global_center[1], c='black', s=120)
plt.scatter(4.6, 0.2, c='black', s=50)#v'
plt.annotate(r'$global\_center$', xy=(global_center[0], global_center[1]), fontsize=15, color='black',
             xytext=(global_center[0]+0.5, global_center[1]-0.2) )


plt.annotate(r'$V_1$', xy=(0, 1), fontsize=15, color='black',
             xytext=(0, 2), )
plt.annotate(r'$\;$', xy=(0, 1), fontsize=13, color='red',
             xytext=(3.9, 3.6), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='black'))

plt.annotate(r'$V_2$', xy=(5, 1), fontsize=15, color='black',
             xytext=(5, 1.5))
plt.annotate(r'$\;$', xy=(5, 1), fontsize=13, color='red',
             xytext=(3.9, 3.6), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='black'))

plt.annotate(r'$V_2^{\prime}$', xy=(2, 1), fontsize=13, color='red',
             xytext=(4.5, -1))
plt.annotate(r'$\;$', xy=(4.6, 0.2), fontsize=13, color='red',
             xytext=(3.9, 3.6), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='red'))

plt.annotate(r'$V_3$', xy=(2.5, 4.5), fontsize=15, color='black',
             xytext=(2.5, 5.5), )
plt.annotate(r'$\;$', xy=(2.5, 4.5), fontsize=13, color='red',
             xytext=(3.9, 3.6), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='black'))

plt.annotate(r'$V_4$', xy=(8, 8), fontsize=15, color='black',
             xytext=(8, 9))
plt.annotate(r'$\;$', xy=(8, 8), fontsize=13, color='red',
             xytext=(3.9, 3.6), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3", color='black'))
plt.show()
