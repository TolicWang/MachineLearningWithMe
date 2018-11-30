# @Time    : 2018/11/29 15:42
# @Email  : wangchengo@126.com
# @File   : ex1.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0


import sys

sys.path.append('../')
from lib.libstring import cut_line
import numpy as np


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    该函数的作用是按行载入数据，然后分词。同时给每个样本构造构造标签
    :param positive_data_file: txt文本格式，其中每一行为一个样本
    :param negative_data_file: txt文本格式，其中每一行为一个样本
    :return:  分词后的结果和标签
    example:
    positive_data_file:
        今天我很高兴，你吃饭了吗？
        这个怎么这么不正式啊？还上进青年
        我觉得这个不错！
    return:
    x_text: [['今天', '我', '很', '高兴'],   ['你', '吃饭', '了', '吗'], ['这个', '怎么', '这么', '不', '正式', '啊', '还', '上进', '青年']]
    y: [1,1,1]
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("载入原始数据......")
    logger.debug("开始清洗数据......")
    positive = []
    negative = []
    for line in open(positive_data_file, encoding='utf-8'):
        positive.append(cut_line(line).split())
    for line in open(negative_data_file, encoding='utf-8'):
        negative.append(cut_line(line).split())
    x_text = positive + negative
    logger.debug("开始构造标签")
    positive_label = [1 for _ in positive]  # 构造标签
    negative_label = [0 for _ in negative]
    y = np.concatenate([positive_label, negative_label], axis=0)

    return x_text, y


def load_word2vec_model(corpus, vector_dir=None, embedding_dim=50, min_count=5, window=7):
    """
    本函数的作用是训练（载入）词向量模型
    :param corpus: 语料，格式为[['A','B','C'],['D','E']] (两个样本)
    :param vector_dir: 路径
    :param embedding_dim: 词向量维度
    :param min_count: 最小词频数
    :param window: 滑动窗口大小
    :return: 训练好的词向量
    """
    import os
    import gensim
    from gensim.models.word2vec import Word2Vec
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("载入词向量模型......")
    if os.path.exists(vector_dir):
        logger.debug("载入已有词向量模型......")
        model = gensim.models.KeyedVectors.load_word2vec_format(vector_dir)
        return model
    logger.debug("开始训练词向量......")
    model = Word2Vec(sentences=corpus, size=embedding_dim, min_count=min_count, window=window)
    model.wv.save_word2vec_format(vector_dir, binary=False)
    return model


def convert_to_vec(sentences, model):
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("转换成词向量......")
    x = np.zeros((len(sentences), model.vector_size))
    for i, item in enumerate(sentences):
        temp_vec = np.zeros((model.vector_size))
        for word in item:
            if word in model.wv.vocab:
                temp_vec += model[word]
        x[i, :] = temp_vec
    return x


def load_dataset(positive_data, negative_data, vec_dir):
    """
    载入数据集
    :param positive_data:
    :param negative_data:
    :param vec_dir:
    :return:
    """
    from sklearn.model_selection import train_test_split
    import logging
    logger = logging.getLogger(__name__)
    logger.info("载入数据集")
    x_text, y = load_data_and_labels(positive_data, negative_data)
    word2vec_model = load_word2vec_model(x_text, vec_dir)
    x = convert_to_vec(x_text, word2vec_model)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20, shuffle=True)
    return X_train, X_test, y_train, y_test


def train():
    from sklearn.tree import DecisionTreeClassifier
    positive_data = './data/email/ham_5000.utf8'
    negative_data = './data/email/spam_5000.utf8'
    vec_dir = './data/vec.model'
    import logging
    logger = logging.getLogger(__name__)
    logger.info("准备中......")
    X_train, X_test, y_train, y_test = load_dataset(positive_data, negative_data, vec_dir)
    model = DecisionTreeClassifier()
    logger.info("开始训练......")
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    train()
