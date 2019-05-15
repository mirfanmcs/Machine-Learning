import numpy as np
import scipy.optimize as opt
import linearRegCostFunction as costReg
import loadData as data

def  trainLinearReg(theta, X, y, iterations, lmda=0.):

    result = opt.minimize(costReg.linearRegCostFunction, theta,args=(X,y, lmda, 1),method='TNC',jac=True,options={'maxiter':iterations})
    thetaOptimized = result.x
    # costOptimized = result.fun

    return thetaOptimized


def main():
    lmda = 0

    # Add 1 as first column to matrix X_norm for xo = 1
    X = np.insert(data.X, 0, 1, axis=1)

    # Set initial &theta; to zero (3x1 vector)
    initial_theta = np.zeros((X.shape[1], 1))

    iterations = 200

    thetaTrained = trainLinearReg(initial_theta, X, data.y, iterations, lmda=lmda)

    print(thetaTrained)
    #Expected value of theta: 13.08790351  0.36777923



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

