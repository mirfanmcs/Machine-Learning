import loadData3 as data3
import matplotlib.pyplot as plot
import numpy as np
import svmTrainGaussian as svmG
from sklearn import svm #SVM software
import plotData as plotData
import plotBoundary as plotBoundary

# DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
# where you select the optimal (C, sigma) learning parameters to use for SVM
# with RBF kernel
#    [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
#    sigma. You should complete this function to return the optimal C and
#    sigma based on a cross-validation set.

def  dataset3Params(X, y, Xval, yval):
	Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
	sigmavalues = Cvalues
	best_pair, best_score = (0, 0), 0

	for Cvalue in Cvalues:
		for sigmavalue in sigmavalues:
			gamma = np.power(sigmavalue, -2.)
			gaus_svm = svm.SVC(C=Cvalue, kernel='rbf', gamma=gamma)
			gaus_svm.fit(X, y.flatten())
			this_score = gaus_svm.score(Xval, yval)
			# print this_score
			if this_score > best_score:
				best_score = this_score
				best_pair = (Cvalue, sigmavalue)


	return best_pair[0], best_pair[1]

def main():
	C, sigma = dataset3Params(data3.X, data3.y, data3.Xval, data3.yval)

	print(C)
	#Expected value: 0.3

	print(sigma)
	#Expecgted value: 0.1

	gaus_svm = svmG.svmTrainGaussian(data3.X, data3.y, C, sigma)

	# Now we plot the decision boundary
	plotData.plotData(plot, data3.X, data3.y)

	plotBoundary.plotBoundary(gaus_svm, -.5,.3,-.8,.6, plot)

	plot.show()



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

