# @Time    : 2019/3/3 14:46
# @Email  : wangchengo@126.com
# @File   : noise.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()
x = data.data
y = data.target

s = 1
for i in range(x.shape[1]):
    for j in range(i + 1, x.shape[1], 1):
        plt.subplot(3, 2, s)
        plt.scatter(x[:, i], x[:, j],c=y)
        s+=1
plt.show()
