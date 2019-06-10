import numpy as np

#Function to draw the SVM boundary
def plotBoundary(my_svm, xmin, xmax, ymin, ymax, plot):
    """
    Function to plot the decision boundary for a trained SVM
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the SVM classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    xvals = np.linspace(xmin,xmax,100)
    yvals = np.linspace(ymin,ymax,100)

    zvals = np.zeros((len(xvals),len(yvals)))

    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[i][j] = float(my_svm.predict(np.asmatrix(np.array([xvals[i],yvals[j]]))))
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plot.contour( xvals, yvals, zvals, [0])
    plot.title("Decision Boundary")

