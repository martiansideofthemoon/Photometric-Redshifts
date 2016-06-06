#!/usr/bin/env python

from keras.models import Sequential
import numpy as np
from math import isnan
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam, RMSprop
import random

TRAINING_SIZE = 10000
TEST_SIZE = 10000
print "Reading data..."
# Training data and corresponding z values
training_data = np.linspace(1, 1000, num=TRAINING_SIZE)
labels = training_data**2
print "Data loaded..."

print "Loading test cases"
test_vector = np.zeros([TEST_SIZE,1])
expected_labels = np.zeros([TEST_SIZE,1])
for i in range(0, TEST_SIZE):
	test_vector[i,0] = random.randrange(0, 1000000)/1000.0
	expected_labels[i,0] = test_vector[i,0]**2
print "test cases loaded"
# Now we define model

print "normalizing"
training_data = training_data / 1000
labels = labels / 1000000
test_vector = test_vector / 1000
expected_labels = expected_labels / 1000000
print "normalized"

print "Defining model..."
model = Sequential()
model.add(Dense(100, input_dim=1, activation='linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid'))
print "Model defined..."

print "Compiling model..."
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
model.compile(loss='mean_squared_error', optimizer=RMSprop())
print "Model compiled..."

model.fit(training_data, labels, nb_epoch = 15, batch_size=100)

print "Predict and plotting..."
print "Mean of test data = " + str(np.mean(test_vector))
print "Median of test data = " + str(np.median(test_vector))

Yp = model.predict(test_vector, batch_size=10000)
assert len(Yp) == len(expected_labels)

# our favorite plot
plt.scatter(expected_labels, Yp)
plt.show()

print "Thank You"
# the end
