import numpy as np

def computeNumericalGradient(J, theta):

    e = 1e-4
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    rows = np.size(theta, 0)

    row_index = 0
    column_index = 0

    for i in range(theta.size):
        if ((row_index + 1)>rows):
            row_index = 0
            column_index = column_index + 1

        perturb[row_index, column_index] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)

        numgrad[row_index, column_index] = (loss2 - loss1)/(2*e)
        perturb[row_index, column_index] = 0

        row_index = row_index + 1

    return numgrad

