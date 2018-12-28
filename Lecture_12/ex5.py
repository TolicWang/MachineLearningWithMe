# @Time    : 2018/12/28 8:37
# @Email  : wangchengo@126.com
# @File   : ex5.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784  # 输入层
OUTPUT_NODE = 10  # 输出层
HIDDEN1_NODE = 512  # 隐藏层
HIDDEN2_NODE = 512  # 隐藏层
BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.6  # 基础学习率
REGULARIZATION_RATE = 0.0001  # 惩罚率
EPOCHES = 50


def inference(input_tensorf):
    w1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, HIDDEN1_NODE], stddev=0.1), dtype=tf.float32, name='w1')
    b1 = tf.Variable(tf.constant(0.0, shape=[HIDDEN1_NODE]), dtype=tf.float32, name='b1')
    a1 = tf.nn.relu(tf.nn.xw_plus_b(input_tensorf, w1, b1))
    tf.add_to_collection('loss', tf.nn.l2_loss(w1))
    w2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_NODE, HIDDEN2_NODE], stddev=0.1), dtype=tf.float32, name='w2')
    b2 = tf.Variable(tf.constant(0.0, shape=[HIDDEN2_NODE]), dtype=tf.float32, name='b2')
    a2 = tf.nn.relu(tf.nn.xw_plus_b(a1, w2, b2))
    tf.add_to_collection('loss', tf.nn.l2_loss(w2))
    w3 = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_NODE, OUTPUT_NODE], stddev=0.1), dtype=tf.float32, name='w3')
    b3 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE], dtype=tf.float32, name='b3'))
    a3 = tf.nn.xw_plus_b(a2, w3, b3)
    tf.add_to_collection('loss', tf.nn.l2_loss(w3))
    return a3


def train():
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name='x_input')
    y_ = tf.placeholder(dtype=tf.int32, shape=[None, OUTPUT_NODE], name='y_input')
    y = inference(x)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    l2_loss = tf.add_n(tf.get_collection('loss'))
    loss = cross_entropy_mean + REGULARIZATION_RATE*l2_loss

    train_op = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE_BASE).minimize(loss=loss)

    prediction = tf.nn.in_top_k(predictions=y, targets=tf.argmax(y_, 1), k=1)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    n_chunk = len(mnist.train.images) // BATCH_SIZE
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCHES):
            for batch in range(n_chunk):
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                feed = {x: batch_xs, y_: batch_ys}
                _, acc, l = sess.run([train_op, accuracy, loss], feed_dict=feed)
                if batch % 50 == 0:
                    print("### Epoch:%d, batch:%d,loss:%.3f, acc on train:%.3f" % (epoch, batch, l, acc))
            if epoch % 5 == 0:
                feed = {x: mnist.test.images, y_: mnist.test.labels}
                acc = sess.run(accuracy, feed_dict=feed)
                print("#### Acc on test:%.3f" % (acc))


if __name__ == '__main__':
    train()
