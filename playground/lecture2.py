"""
Example
"""
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
d = tf.fill([2, 3], x, name='fill')

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))
    print(sess.run(d))

# writer.close()

print(tf.get_default_graph().as_graph_def())

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())
    print(sess.run(assign_op))

p = tf.placeholder(tf.float32, shape=[3])
q = tf.constant([5, 5, 5], tf.float32)
r = p + q
with tf.Session() as sess:
    print(sess.run(r, {p: [6, 6, 6]}))

print(tf.get_default_graph().is_feedable(p))
