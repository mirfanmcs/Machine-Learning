import loadData as data
import gradientDescent as gd
import numpy as np

def predict(X):

    theta = getTheta()
    hx = X * theta

    return hx


def getTheta():
    # Calculate for theta [0,0]
    theta = np.zeros((2,1))
    alpha = 0.01
    iterations = 1500

    theta, J_history = gd.gradientDescent(data.X, data.y, theta, alpha, iterations)

    return theta


def main():
    # Predict for x = 3.5 where x0 = 1 and x1 = 3.5
    x = np.matrix('1, 3.5')
    hx = predict(x)
    print(hx)

    # Predict for x = 7 where x0 = 1 and x1 = 7
    x = np.matrix('1, 7')
    hx = predict(x)
    print(hx)




# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


