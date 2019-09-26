#### <font  color  =red>1.为什么说线性回归中误差是服从均值为0的方差为$\color{red}{\sigma^2}$的正态(高斯)分布，不是0均值行不行？</font>

正态分布：
$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}
$$
对于为什么服从正态分布，参见[最小二乘法与正态分布](https://blog.csdn.net/The_lastest/article/details/82413772)；对于“不是0均值行不行”，答案是行。因为在线性回归中即使均值不为0，我们也可以通过最终通过调节偏置来使得均值为0。

#### <font  color  =red>2.什么是最小二乘法？</font>

预测值与真实值的算术平均值

#### <font  color  =red>3.为什么要用最小二乘法而不是最小四乘法，六乘法？</font>

因为最小二乘法的优化结果，同高斯分布下的极大似然估计结果一样；即最小二乘法是根据基于高斯分布下的极大似然估计推导出来的，而最小四乘法等不能保证这一点。

#### <font  color  =red>4.怎么理解似然函数(likelihood function)</font>

统计学中，似然函数是一种关于统计模型参数的函数。给定输出$X$时，关于参数$\theta$的似然函数$L(\theta|x)$（在数值上）等于给定参数$\theta$后变量$x$的概率：$L(\theta|x)=P(X=x|\theta)$。

统计学的观点始终是认为样本的出现是基于一个分布的。那么我们去假设这个分布为$f$，里面有参数$\theta$。对于不同的$\theta$，样本的分布不一样（例如，质地不同的硬币，即使在大样本下也不可能得出正面朝上的概率相同）。$P(X=x|θ)$表示的就是在给定参数$\theta$的情况下，$x$出现的可能性多大。$L(θ|x)$表示的是在给定样本$x$的时候，哪个参数$\theta$使得$x$出现的可能性多大。所以其实这个等式要表示的核心意思都是在给一个$\theta$和一个样本$x$的时候，整个事件发生的可能性多大。

一句话，对于似然函数就是已知观测结果，但对于不同的分布（不同的参数$\theta$)，将使得出现这一结果的概率不同；

举例：

小明从兜里掏出一枚硬币（质地不均）向上抛了10次，其中正面朝上7次，正面朝下3次；但并不知道在大样本下随机一次正面朝上的概率$\theta$。问：出现这一结果的概率？
$$
P=C_{10}^{7}\theta^{7}(1-\theta)^{3}=120\cdot\theta^{7}(1-\theta)^{3}
$$

```
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,1,500)
y=120*np.power(x,7)*np.power((1-x),3)
plt.scatter(x,y,color='r',linestyle='-',linewidth=0.1)
plt.xlabel(r'$\theta$',fontsize=20)
plt.ylabel('p',fontsize=20)
plt.show()
```

![这里写图片描述](https://img-blog.csdn.net/20180907081312272?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1RoZV9sYXN0ZXN0/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



如图，我们可以发现当且仅当$\theta=0.7$ 时，似然函数取得最大值，即此时情况下事件“正面朝上7次，正面朝下3次”发生的可能性最大，而$\theta=0.7$也就是最大似然估计的结果。

------

**线性回归推导：**

记样本为$(x^{(i)},y^{(i)})$，对样本的观测（预测）值记为$\hat{y}^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}$，则有：
$$
y^{(i)}=\theta^Tx^{(i)}+\epsilon^{(i)}\tag{01}
$$
其中$\epsilon^{(i)}$表示第$i$个预测值与真实值之间的误差，同时由于误差$\epsilon^{(i)}$服从均值为0的高斯分布，于是有：
$$
p(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)}\tag{02}
$$
其中，$p(\epsilon^{(i)})$是概率密度函数

于是将$(1)$带入$(2)$有：
$$
p(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)}\tag{03}
$$
此时请注意看等式$(3)$的右边部分，显然是随机变量$y^{(i)}$，服从以$\theta^Tx^{(i)}$为均值的正态分布（想想正态分布的表达式），又由于该密度函数与参数$\theta,x$有关（即随机变量$(y^{i})$是$x^{(i)},\theta$下的条件分布），于是有：
$$
p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)}\tag{04}
$$
到目前为止，也就是说此时真实值$y^{(i)}$服从均值为$\theta^Tx^{(i)}$,方差为$\sigma^2$的正态分布。同时，由于$\theta^Tx^{(i)}$是依赖于参数$\theta$的变量，那么什么样的一组参数$\theta$能够使得已知的观测值最容易发生呢？此时就要用到极大似然估计来进行参数估计（似然函数的作用就是找到一组参数能够使得随机变量（此处就是$y^{(i)}$）出现的可能性最大）：
$$
L(\theta)=\prod_{i=1}^m p(y^{(i)}|x^{(i)};\theta)=\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)\tag{05}
$$
为了便于求解，在等式$(05)$的两边同时取自然对数：
$$
\begin{aligned}
\log L(\theta)&=\log\left\{ \prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)\right\}\\[3ex]
&=\sum_{i=1}^m\log\left\{\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right)\right\}\\[3ex]
&=\sum_{i=1}^m\left\{\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}\right\}\\[3ex]
&=m\cdot\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\frac{1}{2}\sum_{i=1}^m\left(y^{(i)}-\theta^Tx^{(i)}\right)^2
\end{aligned}
$$
由于$\max L(\theta)\iff\max\log L(\theta)$，所以：
$$
\max\log L(\theta)\iff\min \frac{1}{\sigma^2}\frac{1}{2}\sum_{i=1}^m\left(y^{(i)}-\theta^Tx^{(i)}\right)^2\iff\min\frac{1}{2}\sum_{i=1}^m\left(y^{(i)}-\theta^Tx^{(i)}\right)^2
$$
于是得目标函数：
$$
\begin{aligned}
J(\theta)&=\frac{1}{2m}\sum_{i=1}^m\left(y^{(i)}-\theta^Tx^{(i)}\right)^2\\[3ex]
&=\frac{1}{2m}\sum_{i=1}^m\left(y^{(i)}-Wx^{(i)}\right)^2
\end{aligned}
$$
矢量化：
$$
J = 0.5 * (1 / m) * np.sum((y - np.dot(X, w) - b) ** 2)
$$
**求解梯度**

符号说明：
$y^{(i)}$表示第$i$个样本的真实值；
$\hat{y}^{(i)}$表示第$i$个样本的预测值；
$W$表示权重（列）向量，$W_j$表示其中一个分量；
$X$表示数据集，形状为$m\times n$，$m$为样本个数，$n$为特征维度；
$x^{(i)}$为一个（列）向量，表示第$i$个样本，$x^{(i)}_j$为第$j$维特征
$$
\begin{aligned}
J(W,b)&=\frac{1}{2m}\sum_{i=1}^m\left(y^{(i)}-\hat{y}^{(i)}\right)^2=\frac{1}{2m}\sum_{i=1}^m\left(y^{(i)}-(W^Tx^{(i)}+b)\right)^2\\[4ex]
\frac{\partial J}{\partial W_j}&=\frac{\partial }{\partial W_j}\frac{1}{2m}\sum_{i=1}^m\left(y^{(i)}-(W_1x^{(i)}_1+W_2x^{(i)}_2\cdots W_nx^{(i)}_n+b)\right)^2\\[3ex]
&=\frac{1}{m}\sum_{i=1}^m\left(y^{(i)}-(W_1x^{(i)}_1+W_2x^{(i)}_2\cdots W_nx^{(i)}_n+b)\right)\cdot(-x_j^{(i)})\\[3ex]
&=\frac{1}{m}\sum_{i=1}^m\left(y^{(i)}-(W^Tx^{(i)}+b)\right)\cdot(-x_j^{(i)})\\[4ex]
\frac{\partial J}{\partial b}&=\frac{\partial }{\partial W_j}\frac{1}{2m}\sum_{i=1}^m\left(y^{(i)}-(W^Tx^{(i)}+b)\right)^2\\[3ex]
&=-\frac{1}{m}\sum_{i=1}^m\left(y^{(i)}-(W^Tx^{(i)}+b)\right)\\[3ex]
\frac{\partial J}{\partial W}&=-\frac{1}{m} np.dot(x.T,(y-\hat{y}))\\[3ex]
\frac{\partial J}{\partial b}&=-\frac{1}{m} np.sum(y-\hat{y})\\[3ex]
\end{aligned}
$$

------

#### <font  color  =red>5.怎么理解梯度（Gradient and learning rate），为什么沿着梯度的方向就能保证函数的变化率最大？</font>

首先需要明白梯度是一个向量；其次是函数在任意一点，只有沿着梯度的方向才能保证函数值的变化率最大。

我们知道函数$f(x)$在某点（$x_0$）的导数值决定了其在该点的变化率，也就是说$|f'(x_0)|$越大，则函数$f(x)$在$x=x_0$处的变化速度越快。同时对于高维空间（以三维空间为例）来说，函数$f(x,y)$在某点$(x_0,y_0)$的方向导数值$|\frac{\partial f}{\partial\vec{l}}|$ 的大小还取决于沿着哪个方向求导，也就是说沿着不同的方向，函数$f(x,y)$在$(x_0,y_0)$处的变化率不同。又由于:
$$
\begin{align*}
\frac{\partial f}{\partial\vec{l}}&=\{\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}\} \cdot\{cos\alpha,cos\beta\}\\
&=gradf\cdot\vec{l^0}\\
&=|gradf|\cdot|\vec{l^0}|\cdot cos\theta\\
&=|gradf|\cdot1\cdot cos\theta\\
&=|gradf|\cdot cos\theta
\end{align*}
$$
因此，当$\theta=0$是，即$\vec{l}$与向量（梯度）$\{\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}\}$同向时方向导数取到最大值：
$$\color{red}{\frac{\partial f}{\partial\vec{l}}=|gradf|=\sqrt{(\frac{\partial f}{\partial x})^2+(\frac{\partial f}{\partial y})^2}}$$

故，沿着梯度的方向才能保证函数值的变化率最大。
参见：[方向导数(Directional derivatives)](https://blog.csdn.net/The_lastest/article/details/77898799)、[梯度(Gradient vectors)](https://blog.csdn.net/The_lastest/article/details/77899206)

#### <font  color  =red>6.怎么理解梯度下降算法与学习率（Gradient Descent）？</font>

$$w=w-\alpha\frac{\partial J}{\partial w}$$
梯度下降算法可以看成是空间中的某个点$w$，每次沿着梯度的反方向走一小步，然后更新$w$，然后再走一小步，如此往复直到$J(w)$收敛。而学习率$\alpha$决定的就是在确定方向后每次走多大的“步子”。

#### <font  color  =red>7.学习率过大或者过小将会对目标函数产生什么样的影响？</font>

$\alpha$过大可能会导致目标函数震荡不能收敛，太小则可能需要大量的迭代才能收敛，耗费时间。

#### <font  color  =red>8.运用梯度下降算法的前提是什么？</font>

目标函数为凸函数（形如$y=x^2$）

#### <font  color  =red>9.梯度下降算法是否一定能找到最优解？</font>

对于凸函数而言一定等。对于非凸函数来说，能找到局部最优。

#### 