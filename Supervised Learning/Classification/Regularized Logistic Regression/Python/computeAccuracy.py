import numpy as np
import sigmoid as sig
import optimizeTheta as optTheta
import loadData as data
import mapFeature as map

def computeAccuracy(p):
    accuracy = np.mean(p == data.y) * 100
    return accuracy

def predict(theta, X):
    z = np.dot(X, theta)
    p = np.round(sig.sigmoid(z))
    return p

def getOptimizedTheta():
    thetaOptimized, costOptimized = optTheta.optimizeTheta(lmda=1)

    receive_theta = np.array(thetaOptimized)[np.newaxis]
    theta = np.transpose(receive_theta)
    return theta

def main():
    theta = getOptimizedTheta()
    X_mapped = map.mapFeature(data.X[:, 0], data.X[:, 1])

    p = predict(theta, X_mapped)
    accuracy = computeAccuracy(p)

    print(accuracy)
    #Expected value: 83.05084745762711


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


