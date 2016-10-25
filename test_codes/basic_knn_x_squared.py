import numpy as np
import random
import matplotlib.pyplot

# Test parameters
TRAINING_SIZE = 50
TEST_SIZE = 10000
K = 2


def distance(training_data, test_vector):
	return ((training_data - test_vector)**2)**0.5

# Training data and corresponding z values
training_data = np.linspace(1, 1000, num=TRAINING_SIZE)
labels = training_data**2

# Will contain final points
points = np.zeros([TEST_SIZE, 2])

for i in range(0, TEST_SIZE):
	# A test vector containing five test parameters
	test_vector = random.randrange(0, 1000000)/1000.0
	# Expected z shift value
	expected_label = test_vector**2
	points[i, 1] = expected_label

	# Finding distance of each row of training data with test vector
	v = distance(training_data, test_vector)

	# Taking k nearest neigbours
	v = np.column_stack((labels, v))
	v = v[v[:, 1].argsort()]
	v = v[:K, :]

	# Finding weighted average of x^2 with 1/distance as weights
	points[i, 0] = np.sum(v[:,0]*np.reciprocal(v[:,1]))/np.sum(np.reciprocal(v[:,1]))

matplotlib.pyplot.scatter(points[:,0], points[:,1])
matplotlib.pyplot.show()
