import loadData as data
import numpy as np

def linearRegCostFunction(theta, X, y, lmda, callForOptimization=0):
    # Calculate cost function J of theta J(theta) using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Vectorisation implementation. Applicable for any no for features. 
    # Theta should be a nx1 vector where n = No of feature + 1 

    # Before calling function, append matrix X with 1 for x0=1
    # Vector y is the known value/label of training data

    # Algorithm:
    # J(theta_0,theta_1) = (1/2*m) Sum [i=1 to m] (h(x)-y)^2 + lambda * Sum [i=1 to m] theta ^ 2)
    #     where h(x)=theta_0 + theta_1*x
    #     another way to calculate h(x) is using the matrix. h(x)= x * theta
    #            where theta is 2x1 vector  or transponse of 1x1 matrix

    if callForOptimization == 1:
        receive_theta = np.array(theta)[np.newaxis]
        theta = np.transpose(receive_theta)

    # Do not Regularize theta_0
    tempTheta = np.copy(theta)
    tempTheta[0, 0] = 0



    m = y.size  # number of training examples
    h_x = np.dot(X, theta) # computing h(x)
    sqrErrors = np.square(h_x-y)   # compute square of h(x)-y to compute squre on every item of matrix
    J = (1./(2*m)) * np.sum(sqrErrors,axis=0)   # calculate remaining part of formula (1/2*m) Sum [i=1 to m] (h(x)-y)^2  to the get the computed value of J(theta)

    # Regulisation
    J_Reg  = (lmda / (2. * m)) * np.sum(np.square(tempTheta), axis=0)

    J = J + J_Reg

    temp = np.dot(X, theta)
    error = temp - y

    grad = (1. / m) * np.dot(X.T, error) + ((lmda/m) * tempTheta)

    return J, grad



def main():
    # Calculate for theta [1,1]
    theta = np.matrix('1;1')
    lmda = 1
    # Add 1 as first column to matrix X_norm for xo = 1
    X = np.insert(data.X, 0, 1, axis=1)

    J, grad = linearRegCostFunction(theta, X, data.y, lmda)
    print(J)
    print(grad)
    # Expected value of J: 303.99319222
    # Expected value of grad: -15.30301567, 598.25074417


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()