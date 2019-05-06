import loadData as data
import numpy as np
import predict as pred
#import scipy.misc
import imageio
import utility as util

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

    print(X.shape)

    p = pred.predict(Theta1, Theta2, X)  #calls the neural network prediction method

    if p==10:
        p=0

    return p



def main():

    # Load trained parameters
    Theta1, Theta2 = util.loadTrainedData(data.input_layer_size, data.hidden_layer_size, data.num_labels)

    image = './predict-images/8.png'

    p = predictImg(Theta1, Theta2, image)
    print("Image is number: %d" % p)



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
