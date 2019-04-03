import matplotlib.pyplot as plot
import loadData as data
import numpy as np

def plotData(p, X, y):
    # PLOTDATA Plots the data points X and y into a new figure
    #  plots the data points with + for the positive examples and o for the negative examples.
    #  X is assumed to be a Mx2 matrix.

    # Find Indices of Positive and Negative Examples
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]


    # Plot  data
    p.plot(X[pos, 0], X[pos, 1], 'k+', linewidth=2, markersize=6,label='Admitted')

    p.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=6,label='Not admitted')



    #legend('Admitted', 'Not admitted')

    p.xlabel('Exam 1 score')
    p.ylabel('Exam 2 score')
    p.title('Scatter plot of training data')
    p.legend(loc='upper right')

def main():
    plot.figure()
    plotData(plot, data.X, data.y)
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()



