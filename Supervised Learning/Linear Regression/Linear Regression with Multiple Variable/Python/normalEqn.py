import numpy as np
import loadData as data

def calculateNormalEqn(X, y):
    #NORMALEQN Computes the closed-form solution to linear regression
    #   NORMALEQN(X,y) computes the closed-form solution to linear
    #   regression using the normal equations.

    #Calculate minimum of theta using Normal Equation. This is another method to calculate minimum theta like Gradient Descent

    #Algorithm
    # theta = (X'*X)inv * X' * y


    theta = np.linalg.pinv(X.T * X) * X.T * y

    return theta

def normalEqn():
    # Add 1 as first column to matrix X for xo = 1
    X = np.insert(data.X, 0, 1, axis=1)

    theta = calculateNormalEqn(X, data.y)
    return theta


def main():
    theta = normalEqn()
    print(theta)


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()