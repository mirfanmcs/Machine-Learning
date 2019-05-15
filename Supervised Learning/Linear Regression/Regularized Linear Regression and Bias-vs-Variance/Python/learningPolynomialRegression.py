import numpy as np
import loadData as data
import polyFeatures as pf
import featureNormalize as norm

def learningPolynomialRegression():

	### Degree of ploynomial:
	p = 8

	### Map X onto Polynomial Features and Normalize
	X_poly = pf.polyFeatures(data.X, p)


	# ### Feature Normalize
	X_poly, mu, sigma = norm.featureNormalize(X_poly)


	### Add 1 as first column for x<sub>o</sub> = 1
	X_poly = np.insert(X_poly, 0, 1, axis=1)


	## Map X_poly_test and normalize (using mu and sigma)
	X_poly_test = pf.polyFeatures(data.Xtest, p)
	X_poly_test,_,_ = norm.featureNormalize(X_poly_test, mu=mu, sigma=sigma)


	### Add 1 as first column for x<sub>o</sub> = 1
	X_poly_test = np.insert(X_poly_test, 0, 1, axis=1)

	### Map X_poly_val and normalize (using mu and sigma)
	X_poly_val = pf.polyFeatures(data.Xval, p)
	X_poly_val,_,_ = norm.featureNormalize(X_poly_val, mu=mu, sigma=sigma)

	### Add 1 as first column for x<sub>o</sub> = 1
	X_poly_val = np.insert(X_poly_val, 0, 1, axis=1)


	return X_poly, X_poly_test, X_poly_val, mu, sigma, p





def main():
	X_poly, X_poly_test, X_poly_val, mu, sigma, p = learningPolynomialRegression()
	print(X_poly[0,:])


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
