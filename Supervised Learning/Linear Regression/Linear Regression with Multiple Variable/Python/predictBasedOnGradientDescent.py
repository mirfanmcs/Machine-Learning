import loadData as data
import gradientDescent as gd
import numpy as np
import featureNormalize as fn

def predictBasedOnGradientDescent(X):

    # Normalize
    X_norm = normalize(X)

    # Add 1 as first column to matrix X_norm for xo = 1
    X_norm = np.insert(X_norm, 0, 1., axis=1)

    theta, J_history = gd.gradientDescent()
    hx = X_norm * theta

    return hx


def normalize(X):
    # Call featureNormalize on original data to get value of mu and sigma

    X_norm, mu, sigma = fn.featureNormalize(data.X)

    X[0, 0] = (X[0, 0] - mu[0, 0]) / (sigma[0, 0])
    X[0, 1] = (X[0, 1] - mu[0, 1]) / (sigma[0, 1])

    return X


def main():

    #Estimate the price of a 1650 sq-ft, 3 bedroom house using Gradient Descent.
    X_predict =  np.matrix('1650., 3.')
    hx = predictBasedOnGradientDescent(X_predict)
    print(hx)


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


