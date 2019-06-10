import matplotlib.pyplot as plot
import loadData2 as data
import plotData as plotData
import plotBoundary as plotBoundary
import numpy as np
from sklearn import svm #SVM software

def svmTrainGaussian(X, y, C, sigma):
    gamma = np.power(sigma,-2.)
    gaus_svm = svm.SVC(C, kernel='rbf', gamma=gamma)
    gaus_svm.fit( X, y.flatten())
    return gaus_svm


def main():

    C = 1
    sigma = 0.1

    gaus_svm = svmTrainGaussian(data.X, data.y, C, sigma)

    # Now we plot the decision boundary
    plotData.plotData(plot, data.X, data.y)

    plotBoundary.plotBoundary(gaus_svm, 0, 1, .4, 1.0, plot)

    plot.show()

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
