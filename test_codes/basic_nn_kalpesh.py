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

K = X[:10000,:]
#L = X[100000:110000,:]
M = Y[:10000,:]
#N = Y[100000:110000,:]
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

data = np.column_stack((K, M))
condition = np.amax(data[:,:5], axis=1)-np.amin(data[:,:5], axis=1)<0.05
data1 = data[condition]
data2 = data[np.logical_not(condition)]
data3 = np.amax(data1[:,:5], axis=1)-np.amin(data1[:,:5], axis=1)
data4 = np.amax(data2[:,:5], axis=1)-np.amin(data2[:,:5], axis=1)
print np.mean(data3)
print np.mean(data4)

K = data2[:,:5]
M = data2[:,5]
model.fit(K,M, nb_epoch = 100, batch_size=1000)

print "Predict and plotting..."
Yp = model.predict(K, batch_size=1000)
assert len(Yp) == len(M)

plt.scatter(M, Yp)
plt.show()

print "Thank You"
# the end
