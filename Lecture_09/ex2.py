# @Time    : 2018/11/30 13:30
# @Email  : wangchengo@126.com
# @File   : ex2.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

if __name__ == "__main__":
    import gensim

    model = gensim.models.KeyedVectors.load_word2vec_format('./data/sgns.wiki.word.bz2')
    print('词表长度：', len(model.wv.vocab))
    print('爱    对应的词向量为：', model['爱'])
    print('喜欢  对应的词向量为：', model['喜欢'])
    print('爱  和  喜欢的距离（余弦距离）',model.wv.similarity('爱','喜欢'))
    print('爱  和  喜欢的距离（欧式距离）',model.wv.distance('爱','喜欢'))
    print(model.wv.most_similar(['人类'], topn=3))# 取与给定词最相近的topn个词
    print('爱，喜欢，恨 中最与众不同的是：', model.wv.doesnt_match(['爱', '喜欢', '恨']))
    print(model.wv.doesnt_match(['你','我','他']))#找出与其他词差异最大的词

