#!/usr/bin/env python

import numpy as np
from math import isnan
import matplotlib.pyplot as plt

print "Reading data..."
data = np.genfromtxt("../data/psf2.csv",delimiter=",")[2:,1:]
print "Data loaded..."

print "Filtering data..."
data = np.array(filter(lambda x:((not isnan(x[2])) and min(x[2:])!=-9999 and (not abs(x[1]/x[0])>0.1)), data))
print "Data filtered..."

X = data[:,1:2]
Y = data[:,:1]

plt.scatter(X,Y)
plt.show()

print "Thank You"
# bye
