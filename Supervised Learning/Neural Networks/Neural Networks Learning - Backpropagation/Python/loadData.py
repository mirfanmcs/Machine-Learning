import scipy.io #Used to load the OCTAVE *.mat files

datafile = 'data.mat'
mat = scipy.io.loadmat(datafile)
X, y = mat['X'], mat['y']

m = y.size  #number of training examples

mat = scipy.io.loadmat('weights.mat')
Theta1, Theta2 = mat['Theta1'], mat['Theta2']

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)



