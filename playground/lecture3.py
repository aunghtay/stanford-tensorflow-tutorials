import os
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import xlrd

mpl.use('Agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def figure(d, w, b):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    x, y = d.T[0], d.T[1]
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'bo')
    ax.plot(x, x * w + b, 'r')
    p1 = mpatches.Patch(color='b')
    p2 = mpatches.Patch(color='r')
    fig.legend((p1, p2), ('Real data', 'Predicted data'))
    fig.savefig('lecture3.png')

DATA_FILE = '../examples/data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

Y_predicted = X * w + b

loss = tf.square(Y - Y_predicted, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    for i in range(50):
        total_loss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

    writer.close()
    w, b = sess.run([w, b])

figure(data, w, b)

