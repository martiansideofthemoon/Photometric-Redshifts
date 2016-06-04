import numpy as np
import matplotlib.pyplot
from sklearn import neighbors

# Test parameters
TRAINING_SIZE = 100000
TEST_SIZE = 1000
K = 100


my_data = np.genfromtxt('psf_z.csv', delimiter=',')[1:, :]
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

knn = neighbors.KNeighborsRegressor(K, weights='distance')
output = knn.fit(training_data, labels).predict(test_data)

matplotlib.pyplot.scatter(output, expected_labels)
matplotlib.pyplot.show()
