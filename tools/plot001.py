# @Time    : 2019/3/6 14:28
# @Email  : wangchengo@126.com
# @File   : plot001.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

import matplotlib.pyplot as plt
import numpy as np


rng1=np.random.RandomState(1)
rng2=np.random.RandomState(11)
rng3=np.random.RandomState(21)
x1=rng1.rand(200)*2+1
y1=rng1.rand(200)*10
v1x=np.mean(x1)
v1y=np.mean(y1)


x2=rng2.rand(400)*10+5
y2=rng2.rand(400)*4
v2x=np.mean(x2)
v2y=np.mean(y2)

x3=rng3.rand(600)*10+5
y3=rng2.rand(600)*4+6
v3x=np.mean(x3)
v3y=np.mean(y3)

v0x=(x1.sum()+x2.sum()+x3.sum())/(x1.size+x2.size+x3.size)
v0y=(y1.sum()+y2.sum()+y3.sum())/(y1.size+y2.size+y3.size)
size=20
plt.scatter(x1,y1,c='orange',alpha=0.83,cmap='hsv',s=size)# alpha 控制透明度
plt.scatter(v1x,v1y,c='black',alpha=0.83,cmap='hsv',s=70)

plt.scatter(x2,y2,c='green',alpha=0.83,cmap='hsv',s=size)# alpha 控制透明度
plt.scatter(v2x,v2y,c='black',alpha=0.83,cmap='hsv',s=70)


plt.scatter(x3,y3,c='blue',alpha=0.83,cmap='hsv',s=size)# alpha 控制透明度
plt.scatter(v3x,v3y,c='black',alpha=0.83,cmap='hsv',s=70)

plt.scatter(v0x,v0y,c='black',alpha=0.83,cmap='hsv',s=70)

plt.annotate(r'$z_0=(7.83,5.01)$',xy=(v0x,v0y),fontsize=13,color='black',
             xytext=(v0x+1, v0y), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
plt.annotate(r'$z_1=(2.01,5.01)$',xy=(v1x,v1y),fontsize=13,color='black',
             xytext=(v1x-1, v1y+6), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
plt.annotate(r'$z_2=(10.07,2.03)$',xy=(v2x,v2y),fontsize=13,color='black',
             xytext=(v2x, v2y -3), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
plt.annotate(r'$z_3=(10.07,7.99)$',xy=(v3x,v3y),fontsize=13,color='black',
             xytext=(v3x,v3y+3),arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
plt.xlim(0,16)
plt.ylim(-2,13)
print('V0',v0x,v0y)
print('V1',v1x,v1y)
print('V2',v2x,v2y)
print('V3',v3x,v3y)
plt.show()

