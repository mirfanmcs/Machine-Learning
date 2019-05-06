import numpy as np
import loadData as data
import predict as pred
import utility as util

def computeAccuracy(p):
    accuracy = np.mean(p == data.y.ravel())*100
    return accuracy


def main():
    # Load trained parameters
    Theta1, Theta2 = util.loadTrainedData(data.input_layer_size, data.hidden_layer_size, data.num_labels)

    p = pred.predict(Theta1, Theta2, data.X)

    accuracy = computeAccuracy(p)

    print(accuracy)
    #Expected value: 99.14


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

