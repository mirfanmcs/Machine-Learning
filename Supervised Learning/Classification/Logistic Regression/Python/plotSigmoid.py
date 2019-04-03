import matplotlib.pyplot as plot
import numpy as np
import sigmoid as sig

def  plotSigmoid(p):

    z = np.linspace(-10,10,num=21)

    sig_z = sig.sigmoid(z)

    p.plot(z, sig_z)

    # Add vertical line
    p.plot(np.linspace(0,0,num=21),np.linspace(0,1,num=21),color='k')

    p.xlabel('z')
    p.ylabel('g(z)')
    p.title('Sigmoid')


def main():
    plot.figure()
    plotSigmoid(plot)
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()
