### 1. 本节视频
- 本节视频19,20,21
### 2. 知识点
- 2.1 聚类与无监督算法
- 2.2 聚类与分类的区别
- 2.3 基于距离的聚类算法（Kmeans)
    - [K-Means算法（思想）](https://blog.csdn.net/The_lastest/article/details/78120185)
    - [K-Means算法迭代步骤](Kmeans.md)
    - K值不好确定，且对结果影响较大
    - 初始点的选择对结果影响较大
        - [K-means++算法思想](https://blog.csdn.net/The_lastest/article/details/78288955)
    - 局限性较大，不易发现带有畸形分布簇样本
    - 速度较快
    - Kmeans可视化
        - [Visualizing K-Means Clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)<br>
        <img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0049.png" width="70%"><br>
        <img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0050.png" width="70%"><br>
- 2.4 基于密度的聚类算法（DBSCAN)
    - [DBSCAN算法思想](DBSCAN.md)
    - 不需要指定簇个数
    - 可以发现任意形状的簇
    - 擅长找到离群点（异常点检测）
    - 参数少（但对结果影响大）
    - 数据大时效率低，耗内存
    - DBSCAN可视化
        - [Visualizing DBSCAN Clustering](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)<br>
        <img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0051.png" width="70%"><br>
- 2.5 聚类算法的评估标准
    - 轮廓系数
    - 准确率、召回率
### 3. 示例
  - [手写体聚类分析](ex1.py)
  - [Kmean代码](https://github.com/TolicWang/MachineLearning/blob/master/Cluster/KMeans/Kmeans.py)

### [<主页>](../README.md) [<下一讲>](../Lecture_09/README.md)