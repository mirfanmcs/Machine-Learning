import numpy as np

def  sigmoid(z):

    #SIGMOID Compute sigmoid function
    #   g = SIGMOID(z) computes the sigmoid of z.


    g = 1./(1. + np.exp(-1*z))

    return g

def main():
    print("Sigmoid for 0: %f" % sigmoid(0))
    print("Sigmoid for 1000: %f" % sigmoid(1000))
    print("Sigmoid for -1000: %f" % sigmoid(-1000))


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()

