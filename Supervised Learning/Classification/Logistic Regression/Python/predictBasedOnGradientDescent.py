import gradientDescent as gd
import numpy as np
import sigmoid as sig

def predictBasedOnGradientDescent():
    X = np.matrix('1 45 85')

    theta, J_history = gd.gradientDescent()

    hx = sig.sigmoid(X * theta)
    return hx

def main():
    hx = predictBasedOnGradientDescent()
    print(hx)
    'Expected value: 1'

# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


