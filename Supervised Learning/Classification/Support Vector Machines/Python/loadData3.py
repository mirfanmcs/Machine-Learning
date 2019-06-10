import scipy.io #Used to load the OCTAVE *.mat files

datafile = 'ex6data3.mat'
mat = scipy.io.loadmat(datafile)
X, y, Xval, yval = mat['X'], mat['y'], mat['Xval'], mat['yval']

