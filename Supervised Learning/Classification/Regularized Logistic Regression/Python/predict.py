import numpy as np
import sigmoid as sig
import optimizeTheta as optTheta
import mapFeature as map


def predict():
    X = np.matrix('0.5, 0.3')
    X_mapped = map.mapFeature(X[:, 0], X[:, 1])

    thetaOptimized, costOptimized = optTheta.optimizeTheta(lmda=1)

    z = np.dot(X_mapped ,thetaOptimized)

    hx = sig.sigmoid(z)
    return hx

def main():
    hx = predict()
    print(hx)
    #Expected value: 0.72710959


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


