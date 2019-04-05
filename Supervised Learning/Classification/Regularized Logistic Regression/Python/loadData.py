import numpy as np

datafile = 'data.txt'
data = np.matrix(np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True))
data = np.transpose(data)

# Note: Python is 0 index based so first column is index 0
#In our training data first two columns contains microchip tests on two different tests (X). From these two tests,
# you would like to determine whether the microchips should be accepted or rejected (y).



# X is feature set
X = data[:,0:2]

# Vector y is the known value/label of training data
y = data[:,2]

m = y.size  #number of training examples

