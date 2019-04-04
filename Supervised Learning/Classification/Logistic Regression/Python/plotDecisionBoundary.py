import plotData as pd
import matplotlib.pyplot as plot
import numpy as np
import loadData as data
import optimizeTheta as optTheta


def plotDecisionBoundary(p,theta, X):
    # Plotting the decision boundary: two points, draw a line between
    # Decision boundary occurs when h = 0, or when
    # theta0 + theta1*x1 + theta2*x2 = 0
    # y=mx+b is replaced by x2 = (-1/thetheta2)(theta0 + theta1*x1)

    boundary_X = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    boundary_y = (-1. / theta[2]) * (theta[0] + theta[1] * boundary_X)
    pd.plotData(p, data.X, data.y)
    p.plot(boundary_X, boundary_y, 'b-', label='Decision Boundary')
    p.legend(loc='upper right', fontsize=6)
    p.title('Scatter plot of training data with Decision Boundary')



def main():
    thetaOptimized, costOptimized = optTheta.optimizeTheta()

    plot.figure()
    plotDecisionBoundary(plot, thetaOptimized, data.X)
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
