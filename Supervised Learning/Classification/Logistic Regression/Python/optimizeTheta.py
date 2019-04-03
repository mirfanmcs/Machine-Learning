import numpy as np
import scipy.optimize as opt
import costFunction as cost
import loadData as data



def  calculateOptimizeTheta(theta, X, y, iterations):

    result = opt.minimize(cost.costFunction, theta,args=(X,y, 1),method='TNC',jac=True,options={'maxiter':iterations})
    thetaOptimized = result.x
    costOptimized = result.fun

    return thetaOptimized, costOptimized

def optimizeTheta():
    # Add 1 as first column to matrix X for xo = 1
    X = np.insert(data.X, 0, 1, axis=1)
    y = data.y

    # Set initial &theta; to zero (3x1 vector)
    initial_theta = np.zeros((3, 1))

    iterations = 400

    thetaOptimized, costOptimized = calculateOptimizeTheta(initial_theta, X, y, iterations)
    return thetaOptimized, costOptimized


def main():
    thetaOptimized, costOptimized = optimizeTheta()

    print(costOptimized)
    #Expected value of cost: 0.2034977

    print(thetaOptimized)
    #Expected value of theta: -25.16131854   0.20623159   0.20147149


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

