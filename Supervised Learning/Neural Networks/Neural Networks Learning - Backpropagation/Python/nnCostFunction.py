import utility as util
import loadData as data
import numpy as np
import sigmoid as sig
import numpy.matlib
import sigmoidGradient as sg

# NNCOSTFUNCTION Implements the neural network cost function for a two layer
# neural network which performs classification
#    [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
#    X, y, lambda) computes the cost and gradient of the neural network. The
#    parameters for the neural network are "unrolled" into the vector
#    nn_params and need to be converted back into the weight matrices.
#
#    The returned parameter grad will be a "unrolled" vector of the
#    partial derivatives of the neural network.
#
#
#  Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
#  for our 2 layer neural network


def nnCostFunction(nn_params, X, yin, lmda, input_layer_size, hidden_layer_size, num_labels):

    # Reshape unrolled parameters
    Theta1, Theta2 = util.reshapeParams(nn_params, input_layer_size, hidden_layer_size, num_labels)


    # =============== Implement Forward propagation:
    a1, a2, a3 = propagateForward(nn_params, X, input_layer_size, hidden_layer_size, num_labels)

    m = np.size(X, 0)

    # Setting y in the form of actual output i.e. for '2', vector will be [0 1 0 0 0 0 0 0 0 0]
    # Note: for '0', vector will be [0 0 0 0 0 0 0 0 0 1]
    y = (np.matlib.repmat(np.linspace(1, num_labels, num_labels), m, 1) == np.matlib.repmat(yin, 1, num_labels))*1

    # Calculate cost
    J = -1 * (1 / m) * np.sum( np.sum( (np.multiply(np.log(a3),y) + np.multiply(np.log(1 - a3), (1 - y))) ))

    # Regularization
    error = (lmda / (2 * m)) * ( np.sum( np.sum(np.square(Theta1[:,1:]))) +  np.sum( np.sum(np.square(Theta2[:,1:]))))

    J = J + error

    grad = backPropagate(a1, a2, a3, m, y, Theta1, Theta2, lmda)
    return J, grad

def propagateForward(nn_params, X, input_layer_size, hidden_layer_size, num_labels):
    Theta1, Theta2 = util.reshapeParams(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # Insert bias=1
    a1 = np.insert(X, 0, 1, axis=1)
    a2 = sig.sigmoid(np.dot(a1, Theta1.T))

    # Insert bias=1
    a2 = np.insert(a2, 0, 1, axis=1)
    z2 = np.dot(a2, Theta2.T)
    a3 = sig.sigmoid(z2)

    return [a1, a2, a3]

# def backPropagate(nn_params, X, yin, lmda, input_layer_size, hidden_layer_size, num_labels):
def backPropagate(a1, a2, a3, m, y, Theta1, Theta2, lmda):

    # Theta1, Theta2 = util.reshapeParams(nn_params, input_layer_size, hidden_layer_size, num_labels)
    # a1, a2, a3 = propagateForward(nn_params, X, input_layer_size, hidden_layer_size, num_labels)

    # m = np.size(X, 0)

    # Setting y in the form of actual output i.e. for '2', vector will be [0 1 0 0 0 0 0 0 0 0]
    # Note: for '0', vector will be [0 0 0 0 0 0 0 0 0 1]
    # y = (np.matlib.repmat(np.linspace(1, num_labels, num_labels), m, 1) == np.matlib.repmat(yin, 1, num_labels))*1

    del1 = np.zeros(Theta1.shape)
    del2 = np.zeros(Theta2.shape)


    for t in range(m):
        a1t = np.asmatrix(a1[t, :])
        a2t = np.asmatrix(a2[t, :])
        a3t = np.asmatrix(a3[t, :])
        yt = y[t, :]
        d3 = a3t - yt
        d2 = np.multiply((Theta2.T * d3.T), sg.sigmoidGradient(np.insert((Theta1 * a1t.T), 0, 1, axis=0)))

        del1 = del1 + d2[1:, :] * a1t
        del2 = del2 + d3.T * a2t

    Theta1_grad = 1./m * del1 + (lmda/m) * np.asmatrix(np.concatenate((np.asarray(np.zeros((np.size(Theta1[:,1:], 0), 1))), np.asarray(Theta1[:,1:])), axis=1))
    Theta2_grad = 1./m * del2 + (lmda/m) * np.asmatrix(np.concatenate((np.asarray(np.zeros((np.size(Theta2[:,1:], 0), 1))), np.asarray(Theta2[:,1:])), axis=1))



    # Unroll gradients
    grad = util.unrollParameters(Theta1_grad, Theta2_grad)

    return grad

def main():
    nn_params = util.unrollParameters(data.Theta1,data.Theta2)

    # #========== Calling for Lambda = 0
    lmda = 0

    #----- Calculate cost
    J, grad = nnCostFunction(nn_params, data.X, data.y, lmda, data.input_layer_size, data.hidden_layer_size, data.num_labels)
    print("Calculation for lamda=%d" % lmda)
    print("Cost:", J)
    # Expected Value: 0.2876291651613189

    print("Gradients: \n", grad[:4, ])
    # Expected first 4 values of grad: `0.000061871, 0.000093880, -0.000192594, -0.000168495`



    #========== Calling for Lambda = 1
    lmda = 1

    #----- Calculate cost
    J, grad = nnCostFunction(nn_params, data.X, data.y, lmda, data.input_layer_size, data.hidden_layer_size, data.num_labels)
    print("Calculation for lamda=%d" % lmda)
    print("Cost:", J)
    # Expected Value: 0.38376985909092365

    print("Gradients: \n", grad[:4, ])
    # Expected first 4 values of grad: `0.000061871, 0.000093880, -0.000192594, -0.000168495`

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()



