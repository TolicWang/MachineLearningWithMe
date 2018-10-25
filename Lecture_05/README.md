### 1. 本节视频
- 本节视频13,14,24
### 2. 知识点
- 2.1 算法原理理解
    -   Bagging:并行构造n个模型，每个模型彼此独立；如，RandomForest
    -   Boosting:串行构造模型，下一个模型的提升依赖于训练好的上以个模型；如，AdaBoost,Xgboost
    -   Stacking:第一阶段得出各自结果，第二阶段再用前一阶段结果训练
- 2.2 数据预处理
    -   分析筛选数据特征
    -   缺失值补充（均值，最值）
    -   特征转换<br>
    [用pandas处理缺失值补全及DictVectorizer特征转换](https://blog.csdn.net/The_lastest/article/details/79103386)
    [利用随机森林对特征重要性进行评估](https://blog.csdn.net/The_lastest/article/details/81151986)
- 2.3 Xgboost
    -   安装
        - 方法一：在线安装
        ```python
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ xgboost
        ```
        - 方法二：本地安装
        首先去[戳此处](https://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost)搜索并下载xboost对应安装包<br>
        (cp27表示python2.7,win32表示32位,amd64表示64位)
        ```python
        pip install xgboost-0.80-cp36-cp36m-win_amd64.whl 
        ```
    - 大致原理
### 3. 示例 
- 3.1 本示例先对特征进行人工分析，然后选择其中7个进行训练
    - [示例1](ex1.py)
- 3.2 本示例先对特征进行评估，然后选择其中3个进行训练
    - [示例2](ex2.py)
- 3.3 本示例是以stacking的思想进行训练
    - [示例3](ex3.py)
### 4. 任务
- 4.1 根据所给[数据集001](../DatasetUrl.md)，预测某人是否患有糖尿病；
- 4.2 根据所给[数据集002](../DatasetUrl.md)，预测泰坦尼克号人员生还情况；<br>

    要求：
    - 要求模型预测的准确率尽可能高；
    - 分模块书写代码(比如数据预处理，不同模型的训练要抽象成函数，具体可参见前面例子)；
    
 