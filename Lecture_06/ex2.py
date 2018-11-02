import re
from collections import Counter
import pickle
import sys, os


def load_all_words(data_dir):
    """
    该函数的作用是返回数据集中所有的单词
    :param data_dir:
    :return:

    example:
    (#15 in our series by
    all_words =['in','our','series','by']
    """
    text = open(data_dir).read().replace('\n', '').lower()
    all_words = re.findall('[a-z]+', text)
    return all_words


def get_edit_one_distance(word='at'):
    """
    该函数的作用是得到一个单词，编辑距离为1情况下的所有可能单词（不一定是正确单词）
    :param word:
    :return:
    example:

    word = 'at'
    edit_one={'att', 'aa', 'am', 'ati', 't', 'abt', 'mt', 'aot', 'atu', 'ay', 'aft', 'ac', 'dat', 'ato', 'ft', 'lat',.......}
    """
    n = len(word)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    edit_one = set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
                   [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
                   [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
                   [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion
    return edit_one


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


def train():
    """
    该函数的作用是训练模型，并且保存
    :return:
    """
    data_dir = './data/spellcheck/big.txt'
    all_words = load_all_words(data_dir=data_dir)
    c = Counter()
    for word in all_words:  # 统计词频
        c[word] += 1
    return c


def predict(word):
    """
    该函数的作用是，当用户输入单词不在预料库中是，然后根据预料库预测某个可能词
    :param word: 输入的单词
    :return:

    example:
    word = 'tha'
    the
    """
    model_dir = './data/spellcheck/model.dic'
    model = load_model(model_dir)
    all_words = [w for w in model]
    if word in all_words:
        correct_word = word
    else:
        all_candidates = get_edit_one_distance(word)
        correct_candidates = []
        unique_words = set(all_words)
        max_fre = 0
        correct_word = ""
        for word in all_candidates:
            if word in unique_words:
                correct_candidates.append(word)
        for word in correct_candidates:
            freq = model.get(word)
            if freq > max_fre:
                max_fre = freq
                correct_word = word
        print("所有的候选词：", correct_candidates)
    print("推断词为：", correct_word)


if __name__ == "__main__":
    while True:
        word = input()
        print("输入词为：", word)
        predict(word)
