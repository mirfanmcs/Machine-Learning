import numpy as np
import matplotlib.pyplot as plt
import plotFit as plotfit
import trainLinearReg as train
import learningCurve as lcurve
import learningPolynomialRegression as lpr
import loadData as data

def learningCurveForPolynomial(X_poly, X_poly_val, yval, X, y, lmda, mu, sigma, p, m):

    initial_theta = np.zeros((X_poly.shape[1], 1))

    theta = train.trainLinearReg(initial_theta, X_poly, y, 200, lmda)

    plt.figure(1)
    plt.plot(X, y,'rx',markersize=5,label='Training Data')

    plotfit.plotFit(min(X), max(X), mu, sigma, theta, p, plt)

    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title(r'Polynomial Regression Fit $\lambda$ = %d' % lmda)

    plt.show()

    error_train, error_val = lcurve.learningCurve(X_poly, y, X_poly_val, yval, lmda)
    lcurve.plotLearningCurve(error_train, error_val, plt)

    plt.title(r'Polynomial Regression Learning Curve $\lambda$ = %d' % lmda)

    plt.show()
    
    return theta, error_train, error_val


def main():
    X_poly, X_poly_test, X_poly_val, mu, sigma, p = lpr.learningPolynomialRegression()


    lmda = 0

    theta, error_train, error_val = learningCurveForPolynomial(X_poly, X_poly_val, data.yval,
                                                               data.X, data.y, lmda, mu, sigma, p, data.m)


    print(theta)
    print(error_train)
    print(error_val)

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
