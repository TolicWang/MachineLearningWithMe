# @Time    : 2018/12/28 12:14
# @Email  : wangchengo@126.com
# @File   : ex1.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

import tensorflow as tf
import numpy as np

image_in_man = np.linspace(1, 50, 50).reshape(1, 2, 5, 5)
image_in_tf = image_in_man.transpose(0, 2, 3, 1)
#
weight_in_man = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]).reshape(1, 2, 3, 3)
weight_in_tf = weight_in_man.transpose(2, 3, 1, 0)
print('image in man:')
print(image_in_man)
# print(image_in_tf)
print('weight in man:')
print(weight_in_man)
# #
x = tf.placeholder(dtype=tf.float32, shape=[1, 5, 5, 2], name='x')
w = tf.placeholder(dtype=tf.float32, shape=[3, 3, 2, 1], name='w')
conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')
with tf.Session() as sess:
    r_in_tf = sess.run(conv, feed_dict={x: image_in_tf, w: weight_in_tf})
    r_in_man = r_in_tf.transpose(0, 3, 1, 2)
    print(r_in_man)
