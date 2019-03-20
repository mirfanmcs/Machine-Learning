import loadData as data
import numpy as np
import computeCost as cost
import featureNormalize as nor

def calculateGradientDescent(X, y, theta, alpha, num_iters):
    # Gradient descent is used to minimize cost function J
    # Vectorisation implementation. Applicable for any no for features. 
    # Theta should be a nx1 vector where n = No of feature + 1 

    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Algorithm
    # theta_i= theta_i - alpha * (1/m) Sum [i=1 to m] (h(x)-y)*x_i
    #    where i= training data 


    m = y.size  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(num_iters):

        hx = X * theta  # computing h(x)
        sqrErrors = hx - y
        newX = sqrErrors.T * X
        theta = theta - ((alpha/m) * newX.T)

        # Save the cost J in every iteration
        J_history[iter,0] = cost.computeCost(X, y, theta)

    return theta, J_history
#  End of function

def gradientDescent():
    # Normalize X
    X_norm, mu, sigma = nor.featureNormalize(data.X)

    # Add 1 as first column to matrix X_norm for xo = 1
    X_norm = np.insert(X_norm, 0, 1, axis=1)

    # Calculate for theta [0,0,0]
    theta = np.zeros((3, 1))
    alpha = 0.01
    iterations = 400

    theta, J_history = calculateGradientDescent(X_norm, data.y, theta, alpha, iterations)

    return theta, J_history


def main():

    theta, J_history = gradientDescent()

    print(theta)
    print(J_history)


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()