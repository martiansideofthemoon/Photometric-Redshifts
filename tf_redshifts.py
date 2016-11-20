import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import math
from math import isnan
import numpy as np
import matplotlib.pyplot as plt

def get_data(filename):
    data = np.genfromtxt(filename, delimiter=",")[2:,1:]
    data = np.array(filter(lambda x:((not isnan(x[2])) and min(x[2:])!=-9999), data))
    X = data[:,2:]
    assert X.shape[1] == 5
    Y = data[:,:1]
    minx, maxx = np.min(X), np.max(X)
    miny, maxy = np.min(Y), np.max(Y)
    X = (X-minx)/(maxx-minx)
    Y = (Y-miny)/(maxy-miny)

    training_X = X[:100000,:]
    training_Y = Y[:100000,:]
    #training_X = np.random.rand(100000, 5)
    #training_Y = np.sum(X, axis=1, keepdims=True)

    test_X = X[100000:110000,:]
    test_Y = Y[100000:110000,:]

    return training_X, training_Y, test_X, test_Y

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.variable_scope(layer_name):
        with tf.variable_scope('weights'):
            Weights = tf.get_variable(shape=[in_size, out_size], name='W', initializer=xavier_initializer())
            tf.histogram_summary(layer_name + '/weights', Weights)
        with tf.variable_scope('biases'):
            biases = tf.get_variable(shape=[1, out_size], name='b', initializer=xavier_initializer())
            tf.histogram_summary(layer_name + '/biases', biases)
        with tf.variable_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.histogram_summary(layer_name + '/outputs', outputs)
        return outputs


# Make up some real data
training_X, training_Y, test_X, test_Y = get_data("data/psf2.csv")

# define placeholder for inputs to network
with tf.variable_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 5], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 5, 100, n_layer=1, activation_function=tf.nn.relu)
l2 = add_layer(l1, 100, 30, n_layer=2, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l2, 30, 1, n_layer=3, activation_function=None)

# the error between prediciton and real data
with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.scalar_summary('loss', loss)

with tf.variable_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/", sess.graph)
# important step
sess.run(tf.initialize_all_variables())

batch_size = 1000
num_epoch = 100

steps_per_epoch = len(training_X) / batch_size

for epoch in range(num_epoch):

    for step in range(steps_per_epoch):
        x_data = training_X[step*batch_size:(step+1)*batch_size]
        y_data = training_Y[step*batch_size:(step+1)*batch_size]
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    loss_val = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
    print "Epoch " + str(epoch) + " and loss = " + str(loss_val)
    result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
    writer.add_summary(result, epoch)

steps_per_epoch = len(test_X) / batch_size
total = []
for step in range(0, steps_per_epoch):
    feed_dict = {
      xs: test_X[step*batch_size:(step+1)*batch_size],
      ys: test_Y[step*batch_size:(step+1)*batch_size]
    }
    total.extend(sess.run(prediction, feed_dict=feed_dict)[:,0])
plt.scatter(test_Y[:,0], total)
plt.show()

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs