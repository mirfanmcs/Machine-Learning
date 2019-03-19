import matplotlib.pyplot as plot
import loadData as data
import plotData as plotData
import gradientDescent as gd
import numpy as np

def plot_H_Theta():
    X = data.X
    x = data.X[:,1]
    y = data.y

    theta = getTheta()
    hx = X * theta

    plot.plot(x, hx,label='Linear regression')
    plot.legend(loc='lower right')


def getTheta():
    # Calculate for theta [0,0]
    theta = np.zeros((2,1))
    alpha = 0.01
    iterations = 1500

    theta, J_history = gd.gradientDescent(data.X, data.y, theta, alpha, iterations)

    return theta


def main():
    plot.figure()
    plotData.plotData(plot)
    plot_H_Theta()
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


