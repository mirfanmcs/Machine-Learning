import scipy.io #Used to load the OCTAVE *.mat files

datafile = 'ex6data2.mat'
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y']






