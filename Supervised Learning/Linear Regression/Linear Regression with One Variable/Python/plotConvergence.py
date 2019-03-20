import matplotlib.pyplot as plot
import gradientDescent as gd
import loadData as data
import numpy as np

def plotConvergence(J_history):
    plot.figure()
    plot.plot(range(len(J_history)), J_history, 'bo')
    plot.title(r'Convergence of J($\theta$)')
    plot.xlabel('Number of iterations')
    plot.ylabel(r'J($\theta$)')


def main():
    theta, J_history = gd.gradientDescent(data.X, data.y, np.zeros((2,1)), 0.01, 1500)
    plotConvergence(J_history)
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()



