import matplotlib.pyplot as plot
import loadData as data
import plotData as plotData
import plotBoundary as plotBoundary
from sklearn import svm #SVM software

def svmTrain(X, y, C):
    linear_svm = svm.SVC(C, kernel='linear')
    # Now we fit the SVM to our X matrix (no bias unit)
    linear_svm.fit(X, y.flatten())
    return linear_svm

def main():

    C = 1

    linear_svm = svmTrain(data.X, data.y, C)

    # Now we plot the decision boundary
    plotData.plotData(plot, data.X, data.y)

    plotBoundary.plotBoundary(linear_svm, 0, 4.5, 1.5, 5, plot)

    plot.show()

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
