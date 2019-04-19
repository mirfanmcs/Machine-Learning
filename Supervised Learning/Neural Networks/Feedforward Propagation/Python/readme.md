Supervised Learning - Neural Networks - Feedforward Propagation (PYthon)
===========================================================================================

Run following commands in Python3, and in the local path where all files are present in local Python folder. 

You need to install [Matplotlib](https://matplotlib.org/index.html) which we are using for plotting the data. 

To [install](https://matplotlib.org/users/installing.html) Matplotlib on Mac run following command: 


`$ python3 -m pip install -U matplotlib`


# Goal 
We previously implemented multi-class logistic regression to recognize handwritten digits. However, logistic regression cannot form more complex hypotheses as it is only a linear classifier.

In this part we will implement a neural network to recognize handwritten digits using the same training set as before. The neural network will be able to represent complex models that form non-linear hypotheses. 

we will be using parameters from a neural network that we have already trained. Our goal is to implement the feedforward propagation algorithm to use our weights for prediction. 


# Load Data

We start by first loading and visualizing the dataset.

We will be working with a dataset that contains handwritten digits.

Our  data set in `data.mat` contains 5000 training examples of handwritten digits. The `.mat` format means that the data has been saved in a native Octave/MATLAB matrix format, instead of a text (ASCII) format like a csv-file. These matrices can be read directly into your program by using the load command. After loading, matrices of the correct dimensions and values will appear in your program’s memory. The matrix will already be named, so we do not need to assign names to them.

There are 5000 training examples in `data.mat`, where each training example is a 20x20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20x20 grid of pixels is “unrolled” into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix X. This gives us a 5000x400 matrix X where every row is a training example for a handwritten digit image.

The second part of the training set is a 5000-dimensional vector y that contains labels for the training set. To make things more compatible with Octave/MATLAB indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a “0” digit is labeled as “10”, while the digits “1” to “9” are labeled as “1” to “9” in their natural order.



##  Load Training Data

Module `loadData.py` will be used to load data. We will use this module in other python files. 

## Load trained parameters 

We are going to use network parameters (&Theta;<sup>(1)</sup>, &Theta;<sup>(2)</sup>) which are already trained by us. These are stored in `weights.mat` and will be loaded into Theta1 and Theta2. The parameters have dimensions that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

## Display data calling displayData custom function

We will visualize a subset of the training set. We will randomly selects rows from X. This function maps each row to a 20x20 pixel grayscale image and displays the images together.

Run displayData.py from command prompt:

`$ python3 displayData.py`


![Plot](figures/figure1.png)


# Prediction

Run predict.py from command prompt:

`$ python3 predict.py`

Script will call the predict function which will calculate the prediction values:

Expected first 4 values of pred: `10`, `10`, `10`, `10`

# Training Set Accuracy

Run computeAccuracy.py from command prompt:

`$ python3 computeAccuracy.py`

Script will call the computeAccuracy function which will do following:
* Call `predict` function in `predict.py` module


Expected value: `97.52`


# Test Model 

Call custom function predictImg and pass the  20x20 pixel image. Note that we generated these images in Neural Networks Octave implementation. 
 
 `$ python3 predictImg.py`


Expected value: `8`

Test for all images `0-9`. Node that for `0`, model will return `10` but we will print `0`.



