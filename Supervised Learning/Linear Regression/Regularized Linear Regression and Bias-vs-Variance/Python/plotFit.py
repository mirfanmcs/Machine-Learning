import numpy as np
import polyFeatures as pf
import featureNormalize as norm


def plotFit(min_x, max_x, mu, sigma, theta, p, plot):

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    # Map the X values
    X_poly = pf.polyFeatures(x, p)
    X_poly, _, _ = norm.featureNormalize(X_poly, mu=mu, sigma=sigma)

    ### Add 1 as first column for x<sub>o</sub> = 1
    X_poly = np.insert(X_poly, 0, 1, axis=1)


    hx = np.dot(X_poly, theta).T
    plot.plot(x, hx, '--')

