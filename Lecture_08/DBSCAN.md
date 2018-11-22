## Density-Based Spatial Clustering of Applications with Noise
## 基于密度的聚类算法

**基本概念：**<br>

1.核心对象：若某个点的密度达到算法设定的阈值则其为核心点(即r邻域内点的数量>=minPts)；<br>
2.epsilon-邻域的距离阈值：半径r；<br>
3.直接密度可达：若点p在点q的r邻域内，且q是核心点则称p-q直接密度可达；<br>
<img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0052.png" width="70%"><br>
4.密度可达：若有一个点的序列$q_0,q_1,q_2...q_k$,对任意$q_i,q_{i-1}$是直接密度可达的，则称从$q_0$到$q_k$密度可达（可以理解为具有传递性）；<br>
<img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0053.png" width="70%"><br>
5.密度相连：若从某核心点p出发，点q和点k；<br>
<img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0054.png" width="70%"><br>
6.边界点：属于一个类的非核心点；<br>
<img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0055.png" width="70%"><br>
7.噪音点：不属于任何一个类的点；<br>

**算法步骤：**<br>
```python
step1: 标记所有对象为unvisited;
step2: Do
step3: 随机选择一个unvisited对象p;
step4: 标记p为visited
step5: if p 的epsilon-邻域至少又MinPts个对象
step6:       创建一个新簇C，并把p添加到C中
step7:       令N为p的epsilon邻域中的对象集合
step8:       for N中每个点p
step9:          if p 是unvisited:
step10:             标记p为visited
step11:             if p是核心对象，则把p的epsilon邻域的这些点都添加到N
step12:             如果p还不是任何簇的成员，把p添加到C
step13:      End for
step14:     输出C
step15: else 标记p为噪声
step16: Until所有对象都为visited
```
 


