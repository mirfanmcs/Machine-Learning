import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import axes3d
import loadData as data
import numpy as np
import computeCost as cost
import gradientDescent as gd

def surfacePlot_J_Theta():
    X = data.X
    y = data.y

    theta0_vals, theta1_vals, J_vals = getJtheta(X, y)

    # We need to transpose J_vals before plotting, or else the axes will be flipped
    J_vals = np.transpose(J_vals)

    figure = plot.figure()
    ax = figure.gca(projection='3d')

    # Plot the 3D surface
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=8, cstride=8, alpha=0.3)


    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'J($\theta$)')

    #Get minimum of theta0 and theta1
    theta = getTheta()

    ax.plot(theta[:1], theta[1:],'rx',markersize=5)


def getTheta():
    # Calculate for theta [0,0]
    theta = np.zeros((2, 1))
    alpha = 0.01
    iterations = 1500

    theta, J_history = gd.gradientDescent(data.X, data.y, theta, alpha, iterations)

    return theta


def getJtheta(X,y):

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, num=100)
    theta1_vals = np.linspace(-1, 4, num=100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

    theta = np.zeros((2, 1))


    # Fill out J_vals
    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
            theta[0, 0] = theta0_vals[i]
            theta[1, 0] = theta1_vals[j]
            J_vals[i,j] = cost.computeCost(X, y, theta);


    return theta0_vals, theta1_vals, J_vals




def main():
    surfacePlot_J_Theta()
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()



