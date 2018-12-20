# @Time    : 2018/12/20 13:20
# @Email  : wangchengo@126.com
# @File   : ex2.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

import tensorflow as tf


def load_data():
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import numpy as np
    data = load_boston()
    # print(data.DESCR)# 数据集描述
    X = data.data
    y = data.target
    ss = StandardScaler()
    X = ss.fit_transform(X)
    y = np.reshape(y, (len(y), 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=3)
    return X_train, X_test, y_train, y_test


def linear_regression():
    X_train, X_test, y_train, y_test = load_data()
    x = tf.placeholder(dtype=tf.float32, shape=[None, 13], name='input_x')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None,1], name='input_y')
    w = tf.Variable(tf.truncated_normal(shape=[13, 1], stddev=0.1, dtype=tf.float32, name='weight'))
    b = tf.Variable(tf.constant(value=0, dtype=tf.float32, shape=[1]), name='bias')

    y = tf.matmul(x, w) + b# 预测函数（前向传播）
    loss = 0.5 * tf.reduce_mean(tf.square(y - y_))# 损失函数表达式

    rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_)))

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            feed = {x: X_train, y_: y_train}
            l, r, _ = sess.run([loss, rmse, train_op], feed_dict=feed)
            if i % 20 == 0:
                print("## loss on train: {},rms on train: {}".format(l, r))
        feed = {x: X_test, y_: y_test}
        r = sess.run(rmse, feed_dict=feed)
        print("## RMSE on test:", r)


if __name__ == '__main__':
    linear_regression()
    # X_train, X_test, y_train, y_test = load_data()
    # print(X_test.shape)
    # print(y_test.shape)