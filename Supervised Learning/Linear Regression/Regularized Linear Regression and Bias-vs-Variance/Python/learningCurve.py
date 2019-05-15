import numpy as np
import trainLinearReg as train
import linearRegCostFunction as cost
import loadData as data
import matplotlib.pyplot as plot


# %LEARNINGCURVE Generates the train and cross validation set errors needed
# %to plot a learning curve
# %   [error_train, error_val] = ...
# %       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
# %       cross validation set errors for a learning curve. In particular,
# %       it returns two vectors of the same length - error_train and
# %       error_val. Then, error_train(i) contains the training error for
# %       i examples (and similarly for error_val(i)).

def learningCurve(X, y, Xval, yval, lmda):

	m = data.m

	error_train = np.zeros((m, 1))
	error_val = np.zeros((m, 1))
	initial_theta = np.zeros((X.shape[1], 1))


	for i in range(m):
		indx = i + 1
		Xtrain = X[0:indx, :]
		Ytrain = y[0:indx]

		theta = train.trainLinearReg(initial_theta, Xtrain,Ytrain,200,lmda)
		error_train[i], _ = cost.linearRegCostFunction(theta, Xtrain,Ytrain,0, 1)
		error_val[i], _ = cost.linearRegCostFunction(theta,Xval,yval,0, 1)


	return error_train, error_val

def plotLearningCurve(error_train, error_val, plt):

	plt.plot(np.linspace(1, 12, num=12) , error_train, label='Train')
	plt.plot(np.linspace(1, 12, num=12), error_val, label='Cross Validation')


	plt.title('Learning curve for linear regression')
	plt.xlabel('Number of training examples')
	plt.ylabel('Error')
	plt.legend(loc='upper right')


def main():
	lmda = 0

	# Add 1 as first columnfor xo = 1
	X_b = np.insert(data.X, 0, 1, axis=1)
	Xval_b = np.insert(data.Xval, 0, 1, axis=1)
	error_train, error_val = learningCurve(X_b, data.y, Xval_b, data.yval, lmda)

	plot.figure()
	plotLearningCurve(error_train, error_val,plot)
	plot.show()

	print(error_train)
	print(error_val)

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()