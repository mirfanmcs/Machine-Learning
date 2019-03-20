import loadData as data
import numpy as np
import normalEqn as ne

def predictBasedOnNormalEqn(X):

    theta = ne.normalEqn()
    hx = X * theta
    return hx


def main():

    #Estimate the price of a 1650 sq-ft, 3 bedroom house using Gradient Descent.
    X_predict =  np.matrix('1, 1650., 3.')
    hx = predictBasedOnNormalEqn(X_predict)
    print(hx)


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


