import numpy as np

# RBFKERNEL returns a radial basis function kernel between x1 and x2
#    sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
#    and returns the value in sim


def gaussianKernel(x1, x2, sigma):

    #% Ensure that x1 and x2 are column vectors
    x1 = x1.T.flatten().T
    x2 = x2.T.flatten().T

    sim = 0

    diff = x1 - x2
    sqDiff = np.square(diff)
    sim = np.exp(-np.sum(sqDiff)/(2*(sigma*sigma)))

    return sim

def main():
    x1 = np.matrix('1, 2, 1')
    x2 = np.matrix('0, 4, -1')
    sigma = 2.0
    sim = gaussianKernel(x1, x2, sigma)

    print(sim)

    #Expected value:  0.32465


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

