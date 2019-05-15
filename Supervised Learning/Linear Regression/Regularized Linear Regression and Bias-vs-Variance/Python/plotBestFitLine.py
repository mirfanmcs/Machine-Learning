import matplotlib.pyplot as plot
import loadData as data
import plotData as plotData
import numpy as np
import trainLinearReg as train

def plotBestFitLine(p):
    theta = getTheta()

    X_b = np.insert(data.X, 0, 1, axis=1)

    hx = np.dot(X_b,theta)
    p.plot(data.X, hx)


def getTheta():
    lmda = 0

    # Add 1 as first column to matrix X_norm for xo = 1
    X = np.insert(data.X, 0, 1, axis=1)

    # Set initial &theta; to zero (3x1 vector)
    initial_theta = np.zeros((X.shape[1], 1))

    iterations = 200

    theta = train.trainLinearReg(initial_theta, X, data.y, iterations, lmda=lmda)


    return theta


def main():
    plot.figure()
    plotData.plotData(plot)
    plotBestFitLine(plot)
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


