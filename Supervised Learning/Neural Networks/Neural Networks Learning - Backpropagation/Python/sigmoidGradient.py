import sigmoid as sig
import numpy as np

def sigmoidGradient(z):
    # SIGMOIDGRADIENT returns the gradient of the sigmoid function
    # evaluated at z
    #    g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
    #    evaluated at z. This should work regardless if z is a matrix or a
    #    vector. In particular, if z is a vector or matrix, you should return
    #    the gradient for each element.

    s = sig.sigmoid(z)
    g = np.multiply(s, (1 - s))

    return g

def main():
    g = sigmoidGradient(np.matrix("-1, -0.5, 0, 0.5, 1"))

    print(g)


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()