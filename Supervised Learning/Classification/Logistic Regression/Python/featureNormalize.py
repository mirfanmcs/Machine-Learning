import loadData as data
import numpy as np

def  featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    # Algorithm
        # x = (x-mu) / sigma
        #  where:
        #       x = feature set
        #       mu = mean of feature set
        #       sigma = standard deviation


    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def main():
    X_norm, mu, sigma = featureNormalize(data.X)
    print(X_norm)
    print(mu)
    print(sigma)

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
