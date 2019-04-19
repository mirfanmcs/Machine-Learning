import loadData as data
import numpy as np
import predictOneVsAll as pred
import scipy.misc

import oneVsAll as onevAll


def predictImg(Theta, Img):

    X = scipy.misc.imread(Img)

    X = np.double(X)  # converts it to double
    temp = np.copy(X)   # creates a copy for later use

    X = (X-128)/255  # normalize the features
    X = X * (temp > 0) #return the original 0 values to the X
    X = X.T
    X = np.reshape(X, X.size)
    X = np.array(X)[np.newaxis]


    p = pred.calculatePredictOneVsAll(Theta, X).item()

    if p==10:
        p=0

    return p



def main():

    lmda = 0.1
    num_labels = 10
    theta = onevAll.oneVsAll(num_labels, lmda)


    image = './predict-images/4.png'

    p = predictImg(theta, image)
    print("Image is number: %d" % p)



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
