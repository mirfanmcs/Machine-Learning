import numpy as np
import sigmoid as sig
import optimizeTheta as optTheta
import loadData as data


def computeAccuracy(p):
    accuracy = np.mean(p == data.y) * 100
    return accuracy

def predict(theta, X):
    p = np.round(sig.sigmoid(X * theta))
    return p

def getOptimizedTheta():
    thetaOptimized, costOptimized = optTheta.optimizeTheta()

    receive_theta = np.array(thetaOptimized)[np.newaxis]
    theta = np.transpose(receive_theta)
    return theta

def main():
    theta = getOptimizedTheta()
    X = np.insert(data.X, 0, 1, axis=1)
    p = predict(theta, X)
    accuracy = computeAccuracy(p)

    print(accuracy)
    'Expected value: 89.0'


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


