#!/usr/bin/env python

from keras.models import Sequential
import numpy as np
from math import isnan
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from keras.optimizers import SGD, Adam, RMSprop

print "Reading data..."
data = np.genfromtxt("../data/psf2.csv",delimiter=",")[2:,1:]
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

K = X[:100000,:]
L = X[100000:110000,:]
M = Y[:100000,:]
N = Y[100000:110000,:]
# Now we define model

print "Defining model..."
model = Sequential()

# add a dense layer with 10 nodes
model.add(Dense(100, input_dim=5))
# activation sigmoid
model.add(Activation('linear'))
# layer to output
model.add(Dense(30, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid'))
print "Model defined..."

print "Compiling model..."
model.compile(loss='mean_squared_error', optimizer=RMSprop())
print "Model compiled..."

model.fit(K, M, nb_epoch = 300, batch_size=10000)

print "Predict and plotting..."
Yp = model.predict(L, batch_size=10000)
assert len(Yp) == len(N)

# our favorite plot
plt.scatter(N, Yp)
plt.show()

print "Thank You"
# the end
