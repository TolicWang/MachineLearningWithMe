### 1. 本节参考
   - 1.1 《21项目玩转深度学习》P1-P13
   - 1.2 《白话深度学习与Tensorflow》第三章
### 2. 知识点
 - 2.1 Tensorflow框架简介与安装
 - 2.2 Tensorflow的运行模式
    - [Tensorflow的大致运行模式（思想）和占位符](https://blog.csdn.net/The_lastest/article/details/81052658)
 <img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0074.gif" width="50%"><br>
 - 2.3 Softmax分类器与交叉熵(Cross entropy)
 
### 3. 示例
 - [Tensorflow简单示例](./ex1.py)
 - [基于Tensorflow的波士顿房价预测](./ex2.py)
 - [Tensorflow 两层全连接神经网络拟合正弦函数](./ex3.py)<br>
  <img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0072.png" width="50%"><br>
  <img src ="https://github.com/TolicWang/Pictures/blob/master/Pic/p0073.png" width="50%"><br>
    - [Tensorflow 两层全连接神经网络拟合正弦函数](https://blog.csdn.net/The_lastest/article/details/82848257)
 - [基于Softmax的MNIST手写体识别](./ex4.py)
 
### 4. 作业
- 基于Tensorflow实现一个深层神经网络分类器
    - [参考：TensoFlow全连接网络MNIST数字识别与计算图](https://blog.csdn.net/The_lastest/article/details/81054417)
 
 
### 5. 总结
 - 对于一般的深度学习任务，常见为如下步骤（套路）:
    - (1) 选定模型
    - (2) 定义占位符写出前向传播过程
    - (3) 选定优化方法
    - (4) 定义好回话开始训练
 

### [<主页>](../README.md) [<下一讲>](../Lecture_13/README.md)