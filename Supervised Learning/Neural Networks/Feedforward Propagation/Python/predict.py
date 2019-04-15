import loadData as data
import  sigmoid as sig
import numpy as np

def predict(Theta1, Theta2,X):

    # PREDICT Predict the label of an input given a trained neural network
    #    p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #    trained weights of a neural network (Theta1, Theta2)

    a1 = np.insert(X, 0, 1, axis=1)

    z2 = np.dot(a1, Theta1.T)
    a2 = sig.sigmoid(z2)

    # Insert bias=1
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = np.dot(a2, Theta2.T)

    hypo = sig.sigmoid(z3)

    return (np.argmax(hypo, axis=1) + 1)


def main():
    p = predict(data.Theta1, data.Theta2, data.X)
    print(p[:4, ])
    # Expected first 4 values: 10, 10, 10, 10


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()