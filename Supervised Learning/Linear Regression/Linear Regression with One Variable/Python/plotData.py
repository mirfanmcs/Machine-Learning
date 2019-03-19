import matplotlib.pyplot as plot
import loadData as data

def plotData(p):
    # Plot second column data as first column (x1) has all 1 for x0=1
    x = data.X[:,1]
    y = data.y

    p.plot(x, y,'rx',markersize=5,label='Training Data')
    p.xlabel('Population of City in 10,000s')
    p.ylabel('Profit in $10,000s');


def main():
    plot.figure()
    plotData(plot)
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


