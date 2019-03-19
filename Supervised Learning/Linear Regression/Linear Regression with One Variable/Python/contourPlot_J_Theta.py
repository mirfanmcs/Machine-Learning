import matplotlib.pyplot as plot
import loadData as data
import numpy as np
import surfacePlot_J_Theta as sp

def contourPlot_J_Theta():
    X = data.X
    y = data.y

    theta0_vals, theta1_vals, J_vals = sp.getJtheta(X, y)

    # We need to transpose J_vals before plotting, or else the axes will be flipped
    J_vals = np.transpose(J_vals)

    plot.figure()
    plot.contour(theta0_vals, theta1_vals, J_vals,np.logspace(-2, 3, 20),linewidths=0.75)

    plot.xlabel(r'$\theta_0$')
    plot.ylabel(r'$\theta_1$')

    #Get minimum of theta0 and theta1
    theta = sp.getTheta()
    plot.plot(theta[:1], theta[1:],'rx',markersize=5)


def main():
    contourPlot_J_Theta()
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()



