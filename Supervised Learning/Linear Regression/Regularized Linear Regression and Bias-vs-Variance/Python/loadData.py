import scipy.io #Used to load the OCTAVE *.mat files

datafile = 'data.mat'
mat = scipy.io.loadmat(datafile)
X, y, Xval, yval, Xtest, ytest = mat['X'], mat['y'], mat['Xval'],\
                                 mat['yval'], mat['Xtest'], mat['ytest']

m = y.size  #number of training examples
