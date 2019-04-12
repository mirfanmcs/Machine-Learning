import numpy as np
import oneVsAll as onevAll
import sigmoid as sig
import loadData as data

def calculatePredictOneVsAll(all_theta, X):

    # PREDICT Predict the label for a trained one-vs-all classifier. The labels
    # are in the range 1..K, where K = size(all_theta, 1).
    #   p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    #   for each example in the matrix X. Note that X contains the examples in
    #   rows. all_theta is a matrix where the i-th row is a trained logistic
    #   regression theta vector for the i-th class. You should set p to a vector
    #   of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    #   for 4 examples)

    probs = sig.sigmoid(X.dot(all_theta.T))

    # Adding one because Python uses zero based indexing for the 10 columns (0-9),
    # while the 10 classes are numbered from 1 to 10.
    return (np.argmax(probs, axis=1) + 1)

def predictOneVsAll():
    X = np.insert(data.X, 0, 1, axis=1)
    lmda = 0.1
    num_labels = 10
    theta = onevAll.oneVsAll(num_labels, lmda)
    p = calculatePredictOneVsAll(theta, X)
    return p

def main():
    p = predictOneVsAll()

    print(p[:4, ])
    # Expected first 4 values: 10, 10, 10, 10


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()