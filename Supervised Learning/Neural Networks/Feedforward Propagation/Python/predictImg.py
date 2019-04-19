import loadData as data
import numpy as np
import predict as pred
import scipy.misc
import imageio

def predictImg(Theta1, Theta2, Img):

    # X = scipy.misc.imread(Img)
    X = imageio.imread(Img)

    X = np.double(X)  # converts it to double
    temp = np.copy(X)   # creates a copy for later use

    X = (X-128)/255  # normalize the features
    X = X * (temp > 0) #return the original 0 values to the X
    X = X.T
    X = np.reshape(X, X.size)
    X = np.array(X)[np.newaxis]


    p = pred.predict(Theta1, Theta2, X).item()  #calls the neural network prediction method

    if p==10:
        p=0

    return p



def main():

    image = './predict-images/8.png'

    p = predictImg(data.Theta1, data.Theta2, image)
    print("Image is number: %d" % p)



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
