import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import math
from math import isnan
import numpy as np
#COMMENT: np.random.seed(1337)
import time
import matplotlib.pyplot as plt
#COMMENT: tf.set_random_seed(1337)

def gloret(name, shape):
  return tf.get_variable(name, shape=shape,
    initializer=xavier_initializer())


INPUT_SIZE = 5
NUM_CLASSES = 1
random_seed = 20

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 30, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('num_epoch', 10, 'Epoch size')
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

  tf.set_random_seed(random_seed)
  with tf.variable_scope('inputs'):
    photo_placeholder = tf.placeholder(tf.float32, shape=(batch_size, INPUT_SIZE), name='photometric_data')
    z_placeholder = tf.placeholder(tf.float32, shape=(batch_size), name='redshift')

  # COMMENT: Use Xavier/Glorot initialization; An implementation is available here
  # http://deliprao.com/archives/100
  # TensorFlow also has a xavier initializer:
  # https://www.tensorflow.org/versions/r0.8/api_docs/python/contrib.layers.html#xavier_initializer
  with tf.variable_scope('hidden1'):
    stddev = 1.0 / math.sqrt(float(INPUT_SIZE))
    weights = tf.Variable(gloret('weights', [INPUT_SIZE, hidden_size1]).initialized_value())
    biases = tf.Variable(tf.zeros([hidden_size1]), name='biases')
    hidden1 = tf.matmul(photo_placeholder, weights) + biases

  with tf.variable_scope('hidden2'):
    stddev = 1.0 / math.sqrt(float(hidden_size1))
    weights = tf.Variable(gloret('weights', [hidden_size1, hidden_size2]).initialized_value())
    biases = tf.Variable(tf.zeros([hidden_size2]), name='biases')
    hidden2 = tf.matmul(hidden1, weights) + biases

  with tf.variable_scope('softmax_linear'):
      weights = tf.Variable(gloret('weights', [hidden_size2, NUM_CLASSES]).initialized_value())
      biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                           name='biases')
      logits = tf.matmul(hidden2, weights) + biases

  with tf.variable_scope('sigmoid'):
    logits = tf.sigmoid(logits)

  with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.squared_difference(logits, z_placeholder))
    tf.scalar_summary(loss.op.name, loss)

  with tf.variable_scope('train'):
    # Create the rmsprop optimizer with the given learning rate.
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Create a variable to track the global step.
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss)

  init = tf.initialize_all_variables()
  sess = tf.Session()
  merged = tf.merge_all_summaries()
  writer = tf.train.SummaryWriter("logs/", sess.graph)
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
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      result = sess.run(merged, feed_dict=feed_dict)
      writer.add_summary(result, epoch*steps_per_epoch + step)
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