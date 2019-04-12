import scipy.io #Used to load the OCTAVE *.mat files

datafile = 'data.mat'
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y']

m = y.size  #number of training examples

