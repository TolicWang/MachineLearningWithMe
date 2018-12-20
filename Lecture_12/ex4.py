# @Time    : 2018/12/20 15:05
# @Email  : wangchengo@126.com
# @File   : ex4.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    print(mnist.train.labels[0])
    print(mnist.validation.images[0])


def inference(x):
    w = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, OUTPUT_NODE], stddev=0.1, dtype=tf.float32, name='weight'))
    b = tf.Variable(tf.constant(value=0, dtype=tf.float32, shape=[OUTPUT_NODE]), name='bias')
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    return y


def train():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name='input_x')
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODE], name='input_y')
    logit = inference(x)
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(logit)))
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    correct_prediciotn = tf.equal(tf.argmax(logit, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediciotn, tf.float32))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            feed = {x: batch_xs, y_: batch_ys}
            _, l, acc = sess.run([train_op, loss, accuracy], feed_dict=feed)
            if i % 550 == 0:
                print("Loss on train {},accuracy {}".format(l, acc))
            if i % 5500 == 0:
                feed = {x: mnist.test.images, y_: mnist.test.labels}
                acc = sess.run(accuracy, feed_dict=feed)
                print("accuracy on test ", acc)


if __name__ == '__main__':
    INPUT_NODE = 784
    OUTPUT_NODE = 10
    LEARNING_RATE = 0.01
    train()