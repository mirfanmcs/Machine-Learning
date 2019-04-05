import numpy as np
import sigmoid as sig
import loadData as data
import mapFeature as map

def costFunctionReg(theta, X, y, callForOptimization=0, lmda=0.):
    # %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #   theta as the parameter for regularized logistic regression and the
    #   gradient of the cost w.r.t. to the parameters.
    #
    # Algorithm:
    # J(theta) = (-1/m) ( Sum [i=1 to m] (y*log(h(x))+ (1-y)*log(1-h(x))) + lambda * Sum [i=1 to m] theta ^ 2)
    #    where h(x) = 1 / 1 + e ^ -theta*x

    m = y.size  # number of training examples

    J = 0
    grad = np.zeros(theta.size)


    #We need to transpose theta because optimize function change the shape of theta
    if callForOptimization == 1:
        receive_theta = np.array(theta)[np.newaxis]
        theta = np.transpose(receive_theta)


    # Do not Regularize theta_0
    tempTheta = np.copy(theta)
    tempTheta[0,0] = 0

    z = np.dot(X, theta)

    # Computing h(x)
    h_x = sig.sigmoid(z)   # h(x) = 1 / 1 + e ^ -theta*x


    temp = y.T * np.log(h_x) + (1 - y.T) * np.log(1 - h_x)
    temp2  = (lmda / (2. * m)) * np.sum(np.square(tempTheta), axis=0)

    J = (-1. / m) * temp + temp2

    # Compute the partial derivatives and set grad to the partial
    #   derivatives of the cost w.r.t. each parameter in theta

    temp = sig.sigmoid(z)
    error = temp - y
    grad = (1. / m) * (X.T * error) + ((lmda/m) * tempTheta)

    return J, grad




def main():

    # Map feature
    X_mapped = map.mapFeature(data.X[:, 0], data.X[:, 1])

    y = data.y

    #### CALL FOR LAMBDA = 1

    # Set initial &theta; to zero (3x1 vector)
    initial_theta = np.zeros((X_mapped.shape[1], 1))
    J, grad = costFunctionReg(initial_theta, X_mapped, y,lmda=1.0)

    print(J)
    #Expected value of cost J: 0.69314718

    #### CALL FOR LAMBDA = 10

    # Set initial &theta; to 1 (3x1 vector)
    initial_theta = np.ones((X_mapped.shape[1], 1))
    J, grad = costFunctionReg(initial_theta, X_mapped, y,lmda=10.0)

    print(J)
    # Expected value of cost J: 3.16450933

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()