import numpy as np

datafile = 'data.txt'
data = np.matrix(np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True))
data = np.transpose(data)

# Note: Python is 0 index based so first column is index 0
# X1 refers to the size of the house (in square feet)
# X2 refers to the number of bedrooms
# y refers to the price of the house


# X is feature set
X = data[:,0:2]

# Vector y is the known value/label of training data
y = data[:,2]

m = y.size  #number of training examples


