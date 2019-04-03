import numpy as np
import sigmoid as sig
import optimizeTheta as optTheta


def predictBasedOnAdvancedOptmization():
    X = np.matrix('1 45 85')

    thetaOptimized, costOptimized = optTheta.optimizeTheta()

    receive_theta = np.array(thetaOptimized)[np.newaxis]
    theta = np.transpose(receive_theta)

    hx = sig.sigmoid(X * theta)
    return hx

def main():
    hx = predictBasedOnAdvancedOptmization()
    print(hx)
    'Expected value: 0.77629062'


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


