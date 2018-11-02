def clean_str(string, sep=" "):
    import re
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
    import jieba
    """
    该函数的作用是 先清洗字符串，然后分词
    :param line: 输入必须是字符串类型
    :return: 分词后的结果

    example:
    s ='我今天很高兴'
    print(cut_line(s))# 我 今天 很 高兴
    """
    line = clean_str(line)
    # seg_list = jieba.cut(line)
    # cut_words = " ".join(seg_list)
    cut_words = jieba.lcut(line)
    return cut_words


def drop_stopwords(line, lenth=1):
    return [word for word in line if len(word) > lenth]


def read_data(data_dir=None):
    all_words = []
    for line in open(data_dir, encoding='utf-8'):
        line = cut_line(clean_str(line))
        line = drop_stopwords(line)
        all_words += line
    return all_words


def show_word_cloud(data_dir=None, top_k=None):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import matplotlib
    all_words = read_data(data_dir)
    from collections import Counter
    c = Counter()
    for word in all_words:
        c[word] += 1
    top_k_words = {}
    if top_k:
        for k, v in c.most_common(top_k):
            top_k_words[k] = v
    else:
        top_k_words = c
    matplotlib.rcParams['figure.figsize'] = (20, 10)
    word_cloud = WordCloud(font_path='./data/simhei.ttf', background_color='white', max_font_size=70)
    word_cloud = word_cloud.fit_words(top_k_words)
    plt.imshow(word_cloud)
    plt.show()


if __name__ == "__main__":
    data_dir = './data/email/ham_100.utf8'
    show_word_cloud(data_dir,top_k=200)
