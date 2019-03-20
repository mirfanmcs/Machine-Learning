import loadData as data
import numpy as np
import featureNormalize as nor

def computeCost(X, y, theta):
    # Calculate cost function J of theta J(theta) using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Vectorisation implementation. Applicable for any no for features. 
    # Theta should be a nx1 vector where n = No of feature + 1 

    # Before calling function, append matrix X with 1 for x0=1
    # Vector y is the known value/label of training data

    # Algorithm:
    # J(theta_0,theta_1) = (1/2*m) Sum [i=1 to m] (h(x)-y)^2
    #     where h(x)=theta_0 + theta_1*x
    #     another way to calculate h(x) is using the matrix. h(x)= x * theta
    #            where theta is 2x1 vector  or transponse of 1x1 matrix

   


    m = y.size  # number of training examples
    h_x = X * theta # computing h(x)
    sqrErrors = np.square(h_x-y)   # compute square of h(x)-y to compute squre on every item of matrix
    J = (1./(2*m)) * np.sum(sqrErrors,axis=0)   # calculate remaining part of formula (1/2*m) Sum [i=1 to m] (h(x)-y)^2  to the get the computed value of J(theta)

    return J
#  End of function


def main():

    # Normalize X
    X_norm, mu, sigma = nor.featureNormalize(data.X)

    # Add 1 as first column to matrix X_norm for xo = 1
    X_norm = np.insert(X_norm,0,1,axis=1)


    # Calculate for theta [0,0,0]
    theta = np.zeros((3,1))
    J_theta = computeCost(X_norm,data.y,theta)
    print(J_theta)
    # Expected value: 65591548106.45744

    # Calculate for theta [-1,2,3]
    theta = np.matrix('-1; 2; 3')
    J_theta = computeCost(X_norm,data.y,theta)
    print(J_theta)
    # Expected value: 65591516892.34798

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()