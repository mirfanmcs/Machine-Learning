import utility as util
import numpy as np
import nnCostFunction as cost
import computeNumericalGradient as compGd

# CHECKNNGRADIENTS Creates a small neural network to check the
# backpropagation gradients
#    CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
#    backpropagation gradients, it will output the analytical gradients
#    produced by your backprop code and the numerical gradients (computed
#    using computeNumericalGradient). These two gradient computations should
#    result in very similar values.
#

def checkNNGradients(lmda):

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5


    #We generate some 'random' test data
    Theta1 = util.debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = util.debugInitializeWeights(num_labels, hidden_layer_size)


    # Reusing debugInitializeWeights to generate X
    X = util.debugInitializeWeights(m, input_layer_size - 1)
    y = np.asmatrix(1 + np.arange(1, 1+m) % num_labels).T

    # % Unroll parameters
    nn_params = util.unrollParameters(Theta1, Theta2)

    # Short hand for cost function
    costFunc = lambda p: cost.nnCostFunction(p, X, y, lmda, input_layer_size, hidden_layer_size, num_labels)

    # Compute Gradient
    _, grad = costFunc(nn_params)

    # Compute Numerical Gradient
    numgrad = compGd.computeNumericalGradient(costFunc, nn_params)


    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)


    print('Relative Difference: for lamda=%d:' % lmda, diff)




def main():
    print('If your backpropagation implementation is correct, then \n' \
          'the relative difference will be small (less than 1e-9).')

    lmda=0
    checkNNGradients(lmda)
    #Expected difference for lamda=0: 2.404414918658002e-11

    lmda = 3
    checkNNGradients(lmda)
    # Expected difference for lamda=3: 2.2763406298445233e-11



# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
