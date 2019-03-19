import loadData as data
import numpy as np
import computeCost as cost

# Gradient descent is used to minimize cost function J
# Implementation for single variable/feature

def gradientDescent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Algorithm
    # theta_0= theta_0 - alpha * (1/m) Sum [i=1 to m] (h(x)-y)*x_0
    # theta_1= theta_1 - alpha * (1/m) Sum [i=1 to m] (h(x)-y)*x_1
    #    where x_0=1

    # Vectorization implementation which is more efficient

    m = y.size  # number of training examples
    J_history = np.zeros((num_iters, 1))



    for iter in range(num_iters):
        h_x = X * theta # computing h(x)
        sqrErrors = h_x-y

        theta_0 = theta[0,0]
        theta_1 = theta[1,0]

        temp_0 = theta_0 - alpha * (1./m) * np.sum(np.multiply(sqrErrors,X[:,0]), axis=0)[0,0]
        temp_1 = theta_1 - alpha * (1./m) * np.sum(np.multiply(sqrErrors,X[:,1]), axis=0)[0,0]

        theta[0,0] = temp_0
        theta[1,0] = temp_1

        # Save the cost J in every iteration
        J_history[iter,0] = cost.computeCost(X, y, theta)

    return theta, J_history
#  End of function



def main():
    # Calculate for theta [0,0]
    theta = np.zeros((2,1))
    alpha = 0.01
    iterations = 1500

    theta, J_history = gradientDescent(data.X, data.y, theta, alpha, iterations)

    print(theta)
    print(J_history)


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()