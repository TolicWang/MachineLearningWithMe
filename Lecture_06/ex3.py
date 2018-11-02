import pandas as pd
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import sys, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
import pickle


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


def get_tf_idf(features, top_k=None):
    """
    该函数的作用是得到tfidf特征矩阵
    :param features:
    :param top_k: 取出现频率最高的前top_k个词为特征向量，默认取全部（即字典长度）
    :return:

    example:
    X_test = ['没有 你 的 地方 都是 他乡', '没有 你 的 旅行 都是 流浪 较之']
    IFIDF词频矩阵:
    [[0.57615236 0.57615236 0.         0.40993715 0.         0.40993715]
    [0.         0.         0.57615236 0.40993715 0.57615236 0.40993715]]
    """
    print("================Processing in function: %s()! %s=================" %
          (sys._getframe().f_code.co_name, str(datetime.now())[:19]))
    tfidf_model_dir = './data/sougounews/tfidf.mode'
    if os.path.exists(tfidf_model_dir):
        tfidf = load_model(tfidf_model_dir)
        weight = tfidf.transform(features).toarray()
    else:
        stopwors_dir = './data/stopwords/chinaStopwords.txt'
        stopwords = open(stopwors_dir, encoding='utf-8').read().replace('\n', ' ').split()
        tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", stop_words=stopwords, max_features=top_k)
        weight = tfidf.fit_transform(features).toarray()
        save_model(tfidf_model_dir, tfidf)
    del features
    word = tfidf.get_feature_names()
    print('字典长度为:', len(word))
    return weight


def load_and_cut(data_dir=None):
    """
    该函数的作用是载入原始数据，然后返回处理后的数据
    :param data_dir:
    :return:
    content_seg=['经销商   电话   试驾   订车   憬 杭州 滨江区 江陵','计 有   日间 行 车灯 与 运动 保护 型']
    y = [1,1]
    """
    print("================Processing in function: %s()! %s=================" %
          (sys._getframe().f_code.co_name, str(datetime.now())[:19]))
    names = ['category', 'theme', 'URL', 'content']
    data = pd.read_csv(data_dir, names=names, encoding='utf8', sep='\t')
    data = data.dropna()  # 去掉所有含有缺失值的样本（行）
    content = data.content.values.tolist()
    content_seg = []
    for item in content:
        content_seg.append(cut_line(clean_str(item)))
    # labels = data.category.unique()
    label_mapping = {'汽车': 1, '财经': 2, '科技': 3, '健康': 4, '体育': 5, '教育': 6, '文化': 7, '军事': 8, '娱乐': 9, '时尚': 10}
    data['category'] = data['category'].map(label_mapping)
    y = np.array(data['category'])
    del data,content
    return content_seg, y


def get_train_test(data_dir=None, top_k=None):
    """
    该函数的作用是打乱并划分数据集
    :param data_dir:
    :return:
    """
    print("================Processing in function: %s()! %s=================" %
          (sys._getframe().f_code.co_name, str(datetime.now())[:19]))
    x_train, y_train = load_and_cut(data_dir + 'train.txt')
    x_train = get_tf_idf(x_train, top_k=top_k)
    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, shuffle=True, test_size=0.3)
    return x_train, x_dev, y_train, y_dev


def save_model(model_dir='./', para=None):
    """
    该函数的作用是保存传进来的参数para
    :param model_dir: 保存路径
    :param para:
    :return:
    """
    p = {'model': para}
    temp = open(model_dir, 'wb')
    pickle.dump(p, temp)


def load_model(model_dir='./'):
    """
    该函数的作用是载入训练好的模型，如果不存在则训练
    :param model_dir:
    :return:
    """
    if os.path.exists(model_dir):
        p = open(model_dir, 'rb')
        data = pickle.load(p)
        model = data['model']
    else:
        model = train()
        save_model(model_dir, model)
    return model


def train(data_dir, top_k=None):
    print("================Processing in function: %s()! %s=================" %
          (sys._getframe().f_code.co_name, str(datetime.now())[:19]))
    x_train, x_dev, y_train, y_dev = get_train_test(data_dir, top_k)
    model = MultinomialNB()
    model.fit(x_train, y_train)
    score = model.score(x_dev, y_dev)
    save_model('./data/sougounews/model.m',model)
    print("模型已训练成功，准确率为%s,并已保存！" % str(score))


def eval(data_dir):
    x, y = load_and_cut(data_dir + 'test.txt')
    x_test = get_tf_idf(x)
    model = load_model('./data/sougounews/model.m')
    print("在测试集上的准确率为：%s" % model.score(x_test, y))


if __name__ == "__main__":
    data_dir = './data/sougounews/'
    train(data_dir, top_k=30000)
    eval(data_dir)
