import numpy as np

datafile = 'data.txt'
data = np.matrix(np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True))
data = np.transpose(data)

# Note: Python is 0 index based so first column is index 0
#In our test data first two columns contains the exam scores X, and the third column contains
# the label which will indicate if student will be admitted (y=1) or not admitted (y=0) into college
# based on the exam results in X.


# X is feature set
X = data[:,0:2]

# Vector y is the known value/label of training data
y = data[:,2]

m = y.size  #number of training examples

