import numpy as np

def mapFeature(X1, X2):
    # MAPFEATURE Feature mapping function to polynomial features

    #   MAPFEATURE(X1, X2) maps the two input features
    #   to quadratic features.
    #
    #   Returns a new feature array with more features, comprising of
    #   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    #
    #   Inputs X1, X2 must be the same size


    degrees = 6
    out = np.ones((X1.shape[0], 1))

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = np.power(X1, (i-j))
            term2 = np.power(X2, (j))

            term = (term1 * term2).reshape(term1.shape[0], 1)
            out = np.hstack((out, term))
    return out



