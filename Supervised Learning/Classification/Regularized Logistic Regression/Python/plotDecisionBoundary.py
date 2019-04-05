import matplotlib.pyplot as plot
import numpy as np
import mapFeature as map
import plotData as pd
import optimizeTheta as optTheta
import loadData as data

def plotDecisionBoundary(plt,theta, X, y):
    #PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
    #the decision boundary defined by theta
    #   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
    #   positive examples and o for the negative examples. X is assumed to be
    #   a either
    #   1) Mx3 matrix, where the first column is an all-ones column for the
    #      intercept.
    #   2) MxN, N>3 matrix, where the first column is all-ones



    pd.plotData(plt,X[:,1:3], y)

    # Here is the grid range
    x = np.linspace(-1, 1.5, 50)
    y = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            myfeaturesij = map.mapFeature(np.array([x[i]]), np.array([y[j]]))
            z[i][j] = np.dot(theta, myfeaturesij.T)


    z = z.transpose()
    np.meshgrid(x, y)
    plt.contour(x, y, z, [0])


def main():

    lmda =1
    X_mapped = map.mapFeature(data.X[:, 0], data.X[:, 1])
    thetaOptimized, costOptimized = optTheta.optimizeTheta(lmda=lmda)


    plot.figure()
    plotDecisionBoundary(plot, thetaOptimized,X_mapped, data.y)
    plot.xlabel('Microchip Test 1')
    plot.ylabel('Microchip Test 2')
    plot.title(r'$\lambda$ = %d' % lmda)
    plot.show()



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()