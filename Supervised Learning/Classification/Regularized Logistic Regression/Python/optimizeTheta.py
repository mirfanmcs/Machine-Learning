import numpy as np
import scipy.optimize as opt
import costFunctionReg as costReg
import loadData as data
import mapFeature as map

def  calculateOptimizeTheta(theta, X, y, iterations, lmda=0.):

    result = opt.minimize(costReg.costFunctionReg, theta,args=(X,y, 1,lmda),method='TNC',jac=True,options={'maxiter':iterations})
    thetaOptimized = result.x
    costOptimized = result.fun

    return thetaOptimized, costOptimized


def optimizeTheta(lmda=0.):
    X_mapped = map.mapFeature(data.X[:, 0], data.X[:, 1])

    y = data.y

    # Set initial &theta; to zero (3x1 vector)
    initial_theta = np.zeros((X_mapped.shape[1], 1))

    iterations = 400

    thetaOptimized, costOptimized = calculateOptimizeTheta(initial_theta, X_mapped, y, iterations,lmda=lmda)
    return thetaOptimized, costOptimized


def main():
    thetaOptimized, costOptimized = optimizeTheta(lmda=1)

    print(costOptimized)
    #Expected value of cost: 0.52900273



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

