import numpy as np
import matplotlib.pyplot

TRAINING_SIZE = 10000
TEST_SIZE = 1000
K = 1000

my_data = np.genfromtxt('psf_z.csv', delimiter=',')[1:, :]

training_data = my_data[:TRAINING_SIZE, 3:]
labels = my_data[:TRAINING_SIZE, 1]
points = np.zeros([TEST_SIZE, 2])

for i in range(0, TEST_SIZE):
	if i%10 == 0:
		print i
	test_data = my_data[TRAINING_SIZE+i, 3:]
	expected_labels = my_data[TRAINING_SIZE+i, 1]
	v = np.sum((training_data - test_data)**2, axis=1)**0.5
	v = np.column_stack((labels, v))
	v = v[v[:, 1].argsort()]
	v = v[:K, :]
	points[i, 0] = np.sum(v[:,0]*np.reciprocal(v[:,1]))/np.sum(np.reciprocal(v[:,1]))
	points[i, 1] = expected_labels

matplotlib.pyplot.scatter(points[:,0], points[:,1])

matplotlib.pyplot.show()