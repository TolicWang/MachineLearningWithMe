# @Time    : 2018/12/20 13:17
# @Email  : wangchengo@126.com
# @File   : ex1.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0

import tensorflow as tf

a = tf.constant(value=5, dtype=tf.float32)
b = tf.constant(value=6,dtype=tf.float32)
c = a + b
print(c)
with tf.Session() as sess:
    print(sess.run(c))
