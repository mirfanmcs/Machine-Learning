import numpy as np


def unrollParameters(theta1, theta2):
    nn_params = np.asmatrix(
        np.concatenate((np.asarray(theta1.T.flatten().T), np.asarray(theta2.T.flatten().T)), axis=None)).T

    return nn_params


def reshapeParams(flattened_array,input_layer_size, hidden_layer_size, num_labels):
    Theta1 = flattened_array[:(input_layer_size + 1) * hidden_layer_size] \
        .reshape((hidden_layer_size, input_layer_size + 1),order='F')
    Theta2 = flattened_array[(input_layer_size + 1) * hidden_layer_size:] \
        .reshape((num_labels, hidden_layer_size + 1), order='F')


    return [Theta1, Theta2]

def debugInitializeWeights(fan_out, fan_in):
    # Initialize W using "sin". This ensures that W is always of the same values and will be
    # useful for debugging
    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W


def randInitializeWeights(L_in, L_out):

    W_shape = (L_out, 1 + L_in)
    INIT_EPSILON = 0.12
    W = np.random.rand(*W_shape) * 2 * INIT_EPSILON - INIT_EPSILON

    return W

def loadTrainedData(input_layer_size, hidden_layer_size, num_labels):
    trainedTheta = np.load('./trainedTheta.npy')

    # Reshape unrolled parameters
    Theta1, Theta2 = reshapeParams(trainedTheta, input_layer_size, hidden_layer_size, num_labels)

    return Theta1, Theta2