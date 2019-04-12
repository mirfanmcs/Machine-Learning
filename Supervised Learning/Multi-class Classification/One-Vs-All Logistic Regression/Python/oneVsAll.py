import numpy as np
import loadData as data
import optimizeTheta as optTheta

def oneVsAll(num_labels, lmda):

	# ONEVSALL trains multiple logistic regression classifiers and returns all
	# the classifiers in a matrix all_theta, where the i-th row of all_theta
	# corresponds to the classifier for label i
	#    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
	#    logistic regression classifiers and returns each of these classifiers
	#    in a matrix all_theta, where the i-th row of all_theta corresponds
	#    to the classifier for label i


	X = np.insert(data.X, 0, 1, axis=1)
	y = data.y

	# Set initial &theta; to zero (3x1 vector)
	initial_theta = np.zeros((X.shape[1], 1))
	all_theta = np.zeros((num_labels, X.shape[1]))

	iterations = 50


	for c in np.arange(1, num_labels + 1):
		thetaOptimized, costOptimized = optTheta.calculateOptimizeTheta(initial_theta, X, (y == c)*1, iterations, lmda=lmda)
		all_theta[c - 1,:] = np.transpose(thetaOptimized)

	return all_theta


def main():

    lmda = 0.1
    num_labels = 10
    theta = oneVsAll(num_labels,lmda)
    print(theta[:4,:1])

    # Expected first 4 values: -3.00407911, -2.95000729, -4.88676598, -2.2890568



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()



