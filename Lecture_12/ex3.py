# @Time    : 2018/12/20 13:47
# @Email  : wangchengo@126.com
# @File   : ex3.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def gen_data():
    x = np.linspace(-np.pi, np.pi, 100)
    x = np.reshape(x, (len(x), 1))
    y = np.sin(x)
    return x, y


def inference(x):
    w1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, HIDDEN_NODE], stddev=0.1, dtype=tf.float32), name='w1')
    b1 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[HIDDEN_NODE]))
    a1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
    w2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN_NODE, OUTPUT_NODE], stddev=0.1, dtype=tf.float32), name='w2')
    b2 = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[OUTPUT_NODE]))
    y = tf.matmul(a1, w2) + b2
    return y


def train():
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODE], name='y-input')
    y = inference(x)
    loss = tf.reduce_mean(tf.square(y_ - y))  # 均方误差
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    train_x, train_y = gen_data()
    np.random.seed(200)
    shuffle_index = np.random.permutation(train_x.shape[0])  #
    shuffled_X = train_x[shuffle_index]
    shuffle_y = train_y[shuffle_index]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(train_x, train_y, lw=5, c='r')
    plt.ion()
    plt.show()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            feed_dic = {x: shuffled_X, y_: shuffle_y}
            _, l = sess.run([train_step, loss], feed_dict=feed_dic)
            if (i + 1) % 50 == 0:
                print("### loss on train: ", l)
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                y_pre = sess.run(y, feed_dict={x: train_x})
                lines = ax.plot(train_x, y_pre, c='black')
                plt.pause(0.1)


if __name__ == '__main__':
    INPUT_NODE = 1
    HIDDEN_NODE = 50
    OUTPUT_NODE = 1
    LEARNING_RATE = 0.1
    train()

# x, y = gen_data()
# print(x.shape)
# print(y.shape)
