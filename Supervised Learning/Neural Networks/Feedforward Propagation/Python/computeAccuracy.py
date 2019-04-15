import numpy as np
import loadData as data
import predict as pred


def computeAccuracy(p):
    accuracy = np.mean(p == data.y.ravel())*100
    return accuracy


def main():
    p = pred.predict(data.Theta1, data.Theta2, data.X)

    accuracy = computeAccuracy(p)

    print(accuracy)
    #Expected value: 97.52


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

