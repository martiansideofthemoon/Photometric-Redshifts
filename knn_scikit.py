import numpy as np
import matplotlib.pyplot
from sklearn import neighbors
from sklearn.metrics import mean_squared_error

def knn_regression(K, training_data, labels, test_data, weights='distance'):
	knn = neighbors.KNeighborsRegressor(K, weights=weights)
	output = knn.fit(training_data, labels).predict(test_data)
	return output

def k_vs_rms(START_K, END_K, STEP_K, training_data, labels, test_data, expected_labels, weights='distance'):
	num_points = int((END_K - START_K) / STEP_K) + 1
	points = np.zeros([num_points, 2])
	index = -1
	for K in range(START_K, END_K, STEP_K):
		print "k = " + str(K)
		index += 1
		output = knn_regression(K, training_data, labels, test_data, weights)
		v = np.column_stack((output, expected_labels))
		v = v[~np.isnan(v[:,0]),:]
		RMSE = mean_squared_error(v[:,0], v[:,1])**0.5
		points[index,0] = K
		points[index,1] = RMSE
	if points[-1,0] == 0 and points[-1,1] == 0:
		points = points[:-1,:]
	return points

# Test parameters
TRAINING_SIZE = 500000
TEST_SIZE = 10000
# K=21 works best as found by the k_vs_rms() function
K = 21

my_data = np.genfromtxt('data/psf2.csv', delimiter=',')[1:, :]
# Scaling test parameters
for i in range(3,8):
	vector = my_data[:,i]
	my_data[:,i] = (vector - np.amin(vector)) / (np.amax(vector) - np.amin(vector))

# Training data and corresponding z values
training_data = my_data[:TRAINING_SIZE, 3:]
# Redshift of training data
labels = my_data[:TRAINING_SIZE, 1]
# A test vector containing five test parameters
test_data = my_data[TRAINING_SIZE:TRAINING_SIZE+TEST_SIZE, 3:]
# Expected z shift value
expected_labels = my_data[TRAINING_SIZE:TRAINING_SIZE+TEST_SIZE, 1]

output = knn_regression(K, training_data, labels, test_data)
matplotlib.pyplot.scatter(expected_labels, output)
matplotlib.pyplot.title("Estimated Redshift vs True Redshift (K=21)")
# Add some axis labels.
matplotlib.pyplot.xlabel("True Redshift")
matplotlib.pyplot.ylabel("Estimated Redshift")
matplotlib.pyplot.show()

# Code to optimize K
# points = k_vs_rms(1, 100, 1, training_data, labels, test_data, expected_labels, 'uniform')
# matplotlib.pyplot.scatter(points[:,0], points[:,1])
