import numpy as np
import scipy.optimize as opt
import costFunctionReg as costReg
import loadData as data

def  calculateOptimizeTheta(theta, X, y, iterations, lmda=0.):

    result = opt.fmin_cg(costReg.costFunctionReg, fprime=costReg.costGradient, x0=theta, \
                              args=(X, y, lmda), maxiter=iterations, disp=False, \
                              full_output=True)

    return result[0], result[1]


def optimizeTheta(lmda=0.):

    X = np.insert(data.X,0,1,axis=1)
    y = data.y

    # Set initial &theta; to zero (3x1 vector)
    initial_theta = np.zeros((X.shape[1], 1))

    iterations = 50

    thetaOptimized, costOptimized = calculateOptimizeTheta(initial_theta, X, y, iterations,lmda=lmda)
    return thetaOptimized, costOptimized


def main():
    lmda = 0.1
    thetaOptimized, costOptimized = optimizeTheta(lmda=lmda)

    print(costOptimized)
    #Expected value of cost: 160.39425758157174



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

