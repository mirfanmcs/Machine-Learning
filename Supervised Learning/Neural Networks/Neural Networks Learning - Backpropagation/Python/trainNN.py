import utility as util
import loadData as data
import nnCostFunction as cost
import scipy.optimize as opt
import displayData as plot
import numpy as np

def trainNN(X, y, lmda, input_layer_size, hidden_layer_size, num_labels, iterations):

    #Random initialization: Symmetry breaking
    initial_Theta1 = util.randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = util.randInitializeWeights(hidden_layer_size, num_labels)

    ## Unroll parameters
    initial_nn_params = util.unrollParameters(initial_Theta1, initial_Theta2)

    # Short hand for cost function
    costFunc = lambda p: cost.nnCostFunction(p, X, y, lmda, input_layer_size, hidden_layer_size, num_labels)


    result = opt.minimize(costFunc, x0=initial_nn_params,method='TNC',jac=True,options={'maxiter':iterations})

    # Save trained parameters Theta for later use
    np.save('./trainedTheta.npy', result.x)

    # Reshape unrolled parameters
    Theta1, Theta2 = util.reshapeParams(result.x,input_layer_size, hidden_layer_size, num_labels)

    return Theta1, Theta2

def main():
    lmda = 1
    iterations = 200
    Theta1, Theta2 = trainNN(data.X, data.y, lmda, data.input_layer_size, data.hidden_layer_size, data.num_labels, iterations)

    print(Theta1.shape)
    print(Theta2.shape)

    plot.displayHiddenLayer(Theta1)

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

