import numpy as np

datafile = 'data.txt'
data = np.matrix(np.loadtxt(datafile,delimiter=',',usecols=(0,1),unpack=True))
data = np.transpose(data)

# Note: Python is 0 index based so first column is index 0
# X refers to the population size in 10,000s
# y refers to the profit in $10,000s

# Matrix X is a single feature training data. Append matrix X with 1 for x0=1
X = np.insert(data[:,0],0,1,axis=1)  # Add 1 as first column to matrix 'x' for xo = 1

# Vector y is the known value/label of training data
y = data[:,1]

m = y.size  #number of training examples
