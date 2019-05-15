import numpy as np
import loadData as data

# POLYFEATURES Maps X (1D vector) into the p-th power
#    [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
#    maps each example into its polynomial features where
#    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

def polyFeatures(X, p):
	X_poly = np.asmatrix(np.zeros((X.shape[0], p)))

	for i in range(1,p+1):
		X_poly[:,(i-1)] = np.power(X, i)

	return X_poly


def main():
	p = 8  #degree of polynomial
	X_poly = polyFeatures(data.X, p)
	print(X_poly)


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
