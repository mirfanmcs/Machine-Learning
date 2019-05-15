import numpy as np
import trainLinearReg as train
import linearRegCostFunction as cost
import matplotlib.pyplot as plot
import learningPolynomialRegression as lpr
import loadData as data

# %VALIDATIONCURVE Generate the train and validation errors needed to
# %plot a validation curve that we can use to select lambda
# %   [lambda_vec, error_train, error_val] = ...
# %       VALIDATIONCURVE(X, y, Xval, yval) returns the train
# %       and validation errors (in error_train, error_val)
# %       for different values of lambda. You are given the training set (X,
# %       y) and validation set (Xval, yval).

def validationCurve(X, y, Xval, yval):

	# Selected values of lambda (you should not change this)
	lambda_vec = np.matrix('0; 0.001; 0.003; 0.01; 0.03; 0.1; 0.3; 1; 3; 10')

	error_train = np.zeros((np.size(lambda_vec), 1))
	error_val = np.zeros((np.size(lambda_vec), 1))
	initial_theta = np.zeros((X.shape[1], 1))

	for i in range(lambda_vec.size):
		lmda = lambda_vec.item(i)

		theta = train.trainLinearReg(initial_theta, X, y, 200,lmda)


		error_train[i], _ = cost.linearRegCostFunction(theta, X, y,0, 1)
		error_val[i], _ = cost.linearRegCostFunction(theta,Xval,yval,0, 1)


	return lambda_vec, error_train, error_val


def plotValidationCurve(lambda_vec, error_train, error_val):
	plot.figure()
	plot.plot(lambda_vec , error_train, label='Train')
	plot.plot(lambda_vec, error_val, label='Cross Validation')


	plot.title('Learning curve for linear regression')
	plot.xlabel('Lambda')
	plot.ylabel('Error')
	plot.legend(loc='upper right')

	plot.show()

def main():
	X_poly, X_poly_test, X_poly_val, mu, sigma, p = lpr.learningPolynomialRegression()

	lambda_vec, error_train, error_val = validationCurve(X_poly, data.y, X_poly_val, data.yval)
	plotValidationCurve(lambda_vec, error_train, error_val)

	print(error_train)
	print(error_val)

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
