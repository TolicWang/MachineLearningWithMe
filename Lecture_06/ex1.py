import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import sys
from sklearn.naive_bayes import MultinomialNB


def clean_str(string, sep=" "):
    """
    该函数的作用是去掉一个字符串中的所有非中文字符
    :param string: 输入必须是字符串类型
    :param sep: 表示去掉的部分用什么填充，默认为一个空格
    :return: 返回处理后的字符串

    example:
    s = "祝你2018000国庆快乐！"
    print(clean_str(s))# 祝你 国庆快乐
    print(clean_str(s,sep=""))# 祝你国庆快乐
    """
    string = re.sub(r"[^\u4e00-\u9fff]", sep, string)
    string = re.sub(r"\s{2,}", sep, string)  # 若有空格，则最多只保留2个宽度
    return string.strip()


def cut_line(line):
    """
    该函数的作用是 先清洗字符串，然后分词
    :param line: 输入必须是字符串类型
    :return: 分词后的结果

    example:
    s ='我今天很高兴'
    print(cut_line(s))# 我 今天 很 高兴
    """
    line = clean_str(line)
    seg_list = jieba.cut(line)
    cut_words = " ".join(seg_list)
    return cut_words


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
    x_text: ['今天 我 很 高兴   你 吃饭 了 吗', '这个 怎么 这么 不 正式 啊   还 上 进 青年', '我 觉得 这个 不错']
    y: [1,1,1]
    """
    print("================Processing in function: %s() !=================" % sys._getframe().f_code.co_name)
    positive = []
    negative = []
    for line in open(positive_data_file, encoding='utf-8'):
        positive.append(cut_line(line))
    for line in open(negative_data_file, encoding='utf-8'):
        negative.append(cut_line(line))
    x_text = positive + negative

    positive_label = [1 for _ in positive]  # 构造标签
    negative_label = [0 for _ in negative]

    y = np.concatenate([positive_label, negative_label], axis=0)

    return x_text, y


def get_tf_idf(features):
    """
    该函数的作用是得到tfidf特征矩阵
    :param features:
    :return:

    example:
    X_test = ['没有 你 的 地方 都是 他乡', '没有 你 的 旅行 都是 流浪 较之']
    IFIDF词频矩阵:
    [[0.57615236 0.57615236 0.         0.40993715 0.         0.40993715]
    [0.         0.         0.57615236 0.40993715 0.57615236 0.40993715]]
    """
    print("================Processing in function: %s() !=================" % sys._getframe().f_code.co_name)
    stopwors_dir = './data/stopwords/中文停用词库.txt'
    stopwords = open(stopwors_dir, encoding='utf-8').read().replace('\n', ' ').split()
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", stop_words=stopwords)
    weight = tfidf.fit_transform(features).toarray()
    word = tfidf.get_feature_names()
    print('字典长度为:', len(word))
    return weight


def get_train_test(positive_file, negative_file):
    """
    该函数的作用是打乱并划分数据集
    :param positive_file:
    :param negative_file:
    :return:
    """
    print("================Processing in function: %s() !=================" % sys._getframe().f_code.co_name)
    x_text, y = load_data_and_labels(positive_file, negative_file)
    x = get_tf_idf(x_text)
    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3)
    return X_train, X_test, y_train, y_test


def train(positive_file, negative_file):
    print("================Processing in function: %s() !=================" % sys._getframe().f_code.co_name)
    X_train, X_test, y_train, y_test = get_train_test(positive_file, negative_file)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


if __name__ == "__main__":
    positive_file = './data/email/ham_5000.utf8'
    negative_file = './data/email/spam_5000.utf8'
    train(positive_file, negative_file)
