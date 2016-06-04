#!/usr/bin/env python

from keras.models import Sequential
import numpy as np
from math import isnan
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from keras.optimizers import SGD

print "Reading data..."
data = np.genfromtxt("data.csv",delimiter=",")[2:,1:]
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
assert Y.shape[1] == 1
print "I/O vectors generated..."

print "Normalizing data..."
minx, maxx = np.min(X), np.max(X)
miny, maxy = np.min(Y), np.max(Y)

X = (X-minx)/(maxx-minx)
Y = (Y-miny)/(maxy-miny)
print "Data normalized..."

# Now we define model

print "Defining model..."
model = Sequential()

# add a dense layer with 10 nodes
model.add(Dense(50, input_dim=5))
# activation sigmoid
model.add(Activation('relu'))
# layer to output
model.add(Dense(25, activation = 'tanh'))
model.add(Dense(1, activation = 'tanh'))
print "Model defined..."

print "Compiling model..."
model.compile(optimizer=SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
            loss = 'msle')
print "Model compiled..."

model.fit(X, Y, nb_epoch = 10, batch_size=1000)

print "Predict and plotting..."
Yp = model.predict(X, batch_size=1000)
assert len(Yp) == len(Y)

# our favorite plot
plt.scatter(Y, Yp)
plt.show()

print "Thank You"
# the end
