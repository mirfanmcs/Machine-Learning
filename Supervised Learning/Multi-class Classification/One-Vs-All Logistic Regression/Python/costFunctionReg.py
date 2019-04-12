import numpy as np
import sigmoid as sig


def costFunctionReg(theta, X, y, lmda):
    # %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.
    #
    # Algorithm:
    # J(theta) = (-1/m) ( Sum [i=1 to m] (y*log(h(x))+ (1-y)*log(1-h(x))) + lambda * Sum [i=1 to m] theta ^ 2)
    #    where h(x) = 1 / 1 + e ^ -theta*x

    m = y.size
    h = sig.sigmoid(X.dot(theta))

    J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (lmda / (2 * m)) * np.sum(np.square(theta[1:]))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])


def costGradient(theta, X, y, lmda):
    m = y.size
    h = sig.sigmoid(X.dot(theta.reshape(-1, 1)))

    grad = (1 / m) * X.T.dot(h - y) + (lmda / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return (grad.flatten())


def main():

    initial_theta = np.matrix("-2; -1; 1; 2")
    X = np.insert(np.linspace(1,15,num=15).reshape(3, 5).T/10,0,1,axis=1)
    y = np.matrix("1; 0; 1; 0; 1")

    #### CALL FOR LAMBDA = 3
    J = costFunctionReg(initial_theta, X, y,lmda=3.0)
    grad = costGradient(initial_theta, X, y,lmda=3.0)


    print(J)
    #Expected value of cost J: 2.5348194

    print(grad)
    #Expected value of grad: 0.14656137, -0.54855841, 0.72472227, 1.39800296



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()