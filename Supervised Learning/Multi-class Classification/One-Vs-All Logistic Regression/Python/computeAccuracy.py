import numpy as np
import loadData as data
import predictOneVsAll as predictOnVAll


def computeAccuracy(p):
    accuracy = np.mean(p == data.y.ravel())*100
    return accuracy


def main():
    p = predictOnVAll.predictOneVsAll()
    accuracy = computeAccuracy(p)

    print(accuracy)
    #Expected value: 95.08


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


