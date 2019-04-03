import numpy as np
import sigmoid as sig
import loadData as data

def costFunction(theta, X, y, callForOptimization=0):
    #COSTFUNCTION Compute cost and gradient for logistic regression
    #   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    #   parameter for logistic regression and the gradient of the cost
    #   w.r.t. to the parameters.

    # Algorithm:
    # J(theta) = (-1/m) Sum [i=1 to m] (y*log(h(x))+ (1-y)*log(1-h(x)))
    #    where h(x) = 1 / 1 + e ^ -theta*x
    # Initialize some useful values

    m = y.size  # number of training examples

    J = 0
    grad = np.zeros(theta.size)

    #We need to transpose theta because optimize function change the shape of theta
    if callForOptimization == 1:
        receive_theta = np.array(theta)[np.newaxis]
        theta = np.transpose(receive_theta)

    z = X * theta

    # Computing h(x)
    h_x = sig.sigmoid(z)   # h(x) = 1 / 1 + e ^ -theta*x


    temp = y.T * np.log(h_x) + (1 - y.T) * np.log(1 - h_x)
    J = (-1. / m) * temp

    # Compute the partial derivatives and set grad to the partial
    #   derivatives of the cost w.r.t. each parameter in theta

    temp = sig.sigmoid(z)
    error = temp - y
    grad = (1. / m) * (X.T * error)

    return J, grad





def main():

    # Add 1 as first column to matrix X_norm for xo = 1
    X = np.insert(data.X, 0, 1, axis=1)
    y = data.y

    # Set initial &theta; to zero (3x1 vector)
    theta = np.zeros((3, 1))
    J, grad = costFunction(theta, X, y)

    print(J)
    #Expected value of cost J: 0.69314718

    print(grad)
    #Expected value of gradient:  -0.10000, -12.00921659, -11.26284221

    # Call for another values of theta
    theta = np.matrix('-24; 0.2; 0.2')

    J, grad = costFunction(theta, X, y)

    print(J)
    #Expected value of cost J: 0.21833019

    print(grad)
    #Expected value of gradient:  0.04290299, 2.56623412, 2.64679737


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()