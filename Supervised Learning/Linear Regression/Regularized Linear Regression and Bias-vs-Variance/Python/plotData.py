import matplotlib.pyplot as plot
import loadData as data

def plotData(p):

    p.plot(data.X, data.y,'rx',markersize=5,label='Training Data')
    p.xlabel('Change in water level (x)')
    p.ylabel('Water flowing out of the dam (y)');



def main():
    plot.figure()
    plotData(plot)
    plot.show()


# If this script is executed, then main() will be executed
if __name__ == '__main__':
    main()


