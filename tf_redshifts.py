import tensorflow as tf
import math
from math import isnan
import numpy as np
import time
import matplotlib.pyplot as plt

INPUT_SIZE = 5
NUM_CLASSES = 1

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 30, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('num_epoch', 40, 'Epoch size')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')

batch_size = FLAGS.batch_size
hidden_size1 = FLAGS.hidden1
hidden_size2 = FLAGS.hidden2
learning_rate = FLAGS.learning_rate
num_epoch = FLAGS.num_epoch

print "Reading data..."
data = np.genfromtxt("data/psf2.csv",delimiter=",")[2:,1:]
print "Data loaded..."
print "Filtering data..."
data = np.array(filter(lambda x:((not isnan(x[2])) and min(x[2:])!=-9999), data))
print "Data filtered..."
print "Generating I/O vectors..."
# the parameters
X = data[:,2:]
# the following is TMP
assert X.shape[1] == 5
# the output
Y = data[:,:1]

print "Normalizing data..."
minx, maxx = np.min(X), np.max(X)
miny, maxy = np.min(Y), np.max(Y)

X = (X-minx)/(maxx-minx)
Y = (Y-miny)/(maxy-miny)
print "Data normalized..."

training_X = X[:100000,:]
training_Y = Y[:100000,:]

test_X = X[100000:110000,:]
test_Y = Y[100000:110000,:]

with tf.Graph().as_default():

  photo_placeholder = tf.placeholder(tf.float32, shape=(batch_size, INPUT_SIZE))
  z_placeholder = tf.placeholder(tf.float32, shape=(batch_size))

  with tf.name_scope('hidden1'):
    stddev = 1.0 / math.sqrt(float(INPUT_SIZE))
    weights = tf.Variable(tf.truncated_normal([INPUT_SIZE, hidden_size1], stddev=stddev), name='weights')
    biases = tf.Variable(tf.zeros([hidden_size1]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(photo_placeholder, weights) + biases)

  with tf.name_scope('hidden2'):
    stddev = 1.0 / math.sqrt(float(hidden_size1))
    weights = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size2], stddev=stddev), name='weights')
    biases = tf.Variable(tf.zeros([hidden_size2]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope('softmax_linear'):
      weights = tf.Variable(
          tf.truncated_normal([hidden_size2, NUM_CLASSES],
                              stddev=1.0 / math.sqrt(float(hidden_size2))),
          name='weights')
      biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                           name='biases')
      logits = tf.matmul(hidden2, weights) + biases

  with tf.name_scope('sigmoid'):
    logits = tf.sigmoid(logits)

  loss = tf.reduce_mean(tf.squared_difference(logits, z_placeholder))

  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the rmsprop optimizer with the given learning rate.
  optimizer = tf.train.RMSPropOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)

  init = tf.initialize_all_variables()
  sess = tf.Session()

  sess.run(init)

  steps_per_epoch = len(training_X) / batch_size
  print str(steps_per_epoch) + " steps per epoch"

  for epoch in range(0, num_epoch):
    start_time = time.time()
    for step in range(0, steps_per_epoch):
      feed_dict = {
        photo_placeholder: training_X[step*batch_size:(step+1)*batch_size],
        z_placeholder: training_Y[step*batch_size:(step+1)*batch_size,0]
      }
      _, loss_value, biases1 = sess.run([train_op, loss, biases], feed_dict=feed_dict)
    duration = time.time() - start_time
    print('Epoch %d: loss = %.5f (%.3f sec)' % (epoch, loss_value, duration))

  steps_per_epoch = len(test_X) / batch_size
  total = []
  for step in range(0, steps_per_epoch):
    feed_dict = {
      photo_placeholder: test_X[step*batch_size:(step+1)*batch_size],
      z_placeholder: test_Y[step*batch_size:(step+1)*batch_size,0]
    }
    total.extend(sess.run(logits, feed_dict=feed_dict)[:,0])
  plt.scatter(test_Y[:,0], total)
  plt.show()
  #print "Final loss is " + str(math.sqrt(total/10000.0))