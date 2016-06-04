#!/usr/bin/env python

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

filename = "photz.csv"
data = np.genfromtxt(filename, delimiter=',')

# skip 1
data = data[1:]

####### to show data
plt.scatter(data[:,5],data[:,1])
plt.show()
#######

# to plot error
#err = (data[:,1]-data[:,5])/(np.sqrt(2))
#mu, std = norm.fit(err)
#plt.hist(err, bins=25, normed=True, alpha=0.6, color='g')

#xmin, xmax = plt.xlim()
#x = np.linspace(xmin, xmax, 100)
#p = norm.pdf(x, mu, std)
#plt.plot(x,p,'k',linewidth=2)
#
#plt.show()


###### things which show confidence intervals
#conf_int =  norm.interval(0.99, loc=0, scale=std/(np.sqrt(len(err))))
#print conf_int, mu
# the above is sampling statistics for mean of data
######
