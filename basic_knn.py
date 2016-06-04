import numpy as np
import matplotlib.pyplot

# Test parameters
TRAINING_SIZE = 100000
TEST_SIZE = 1000
K = 100


def distance(training_data, test_vector):
	return np.sum((training_data - test_vector)**2, axis=1)**0.5

my_data = np.genfromtxt('psf_z.csv', delimiter=',')[1:, :]
# Scaling test parameters
for i in range(3,8):
	vector = my_data[:,i]
	my_data[:,i] = (vector - np.amin(vector)) / (np.amax(vector) - np.amin(vector))

# Training data and corresponding z values
training_data = my_data[:TRAINING_SIZE, 3:]
# Redshift of training data
labels = my_data[:TRAINING_SIZE, 1]

# Will contain final points
points = np.zeros([TEST_SIZE, 2])

for i in range(0, TEST_SIZE):
	if i%100 == 0:
		print i
	# A test vector containing five test parameters
	test_vector = my_data[TRAINING_SIZE+i, 3:]
	# Expected z shift value
	expected_label = my_data[TRAINING_SIZE+i, 1]
	points[i, 1] = expected_label

	# Finding distance of each row of training data with test vector
	v = distance(training_data, test_vector)

	# Taking k nearest neigbours
	v = np.column_stack((labels, v))
	v = v[v[:, 1].argsort()]
	v = v[:K, :]

	# Finding weighted average of redshift with 1/distance as weights
	points[i, 0] = np.sum(v[:,0]*np.reciprocal(v[:,1]))/np.sum(np.reciprocal(v[:,1]))

matplotlib.pyplot.scatter(points[:,0], points[:,1])
matplotlib.pyplot.show()
