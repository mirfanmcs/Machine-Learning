Supervised Learning - Regularized Linear Regression and Bias-vs-Variance (Octave)
==================================================================================

Note: You can run below code in either Octave or Matlab.

Octave is a free software for mathematics and plotting. You can install Octave from [here](https://www.gnu.org/software/octave/).


Run following commands in Octave, and in the local path where all files are present in local Octave folder. 


`$ octave`

Once in Octave prompt, change the prompt to `>>>` using below command:

`PS1('>>>')`

# Goad
We will implement regularized linear regression and use it to study models with different bias-variance properties. First, we will implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir. Next, we will go through some diagnostics of debugging learning algorithms and examine the effects of bias v.s. variance.


# Load Data

Out dataset contains historical records on the change in the water level, `X`, and the amount of water flowing out of the dam, `y`.

This dataset is divided into three parts:
* A training set that model will learn on: `X`, `y`
* A cross validation set for determining the regularization parameter: `Xval`, `yval`
* A test set for evaluating performance. These are “unseen” examples which your model did not see during training: `Xtest`, `ytest`

## Initialization

Clear all variables 
`>>> clear`

Close all plot windows `>>> close all`

Clear command window/screen `>>> clc`

Load data 

`>>> load('data.mat');`

Load will set variables `X`, `y`, `Xval`,`yval`, `Xtest`,`ytest` with loaded data.

Number of examples:

`>>> m = size(X, 1);`

# Plot training data

`>>> plot(X, y, 'rx', 'MarkerSize', 5, 'LineWidth', 1.5);`

`>>> xlabel('Change in water level (x)');`

`>>> ylabel('Water flowing out of the dam (y)');`


![Plot](figures/figure1.png)


# Regularized Linear Regression Cost 

We will use the linear regression cost function we implemneted before and regularized it with &lambda;.

## Call linearRegCostFunction funciton with intial &theta; of (1,1).

### Set initial value of &theta;

`>>> theta = [1 ; 1];`

### Add 1 as first column to matrix 'X' for x<sub>o</sub> = 1 

`>>> X_b = [ones(m, 1) X]`

### Call cost function for &lambda;=1

`>>> [J, grad] = linearRegCostFunction(X_b, y, theta, 1);`

Expected value of J: `303.99`

Expected value of gradient: `-15.303`, `598.251`


# Train Model - Advanced Optimization

Call `trainLinearReg` function to train model. Function uses advanced optimisation `fmincg` function to train the model. 

## Train linear regression with lambda = 0

`>>> lambda = 0;`

### Add 1 as first column to matrix 'X' for x<sub>o</sub> = 1 

`>>> X_b = [ones(m, 1) X];`

`>>> [theta] = trainLinearReg(X_b, y, lambda);`

 Expected value of &theta;: `13.08790`, `0.36778`


## Plot fit over the data

`>>> plot(X, y, 'rx', 'MarkerSize', 5, 'LineWidth', 1.5);`

`>>> xlabel('Change in water level (x)');`

`>>> ylabel('Water flowing out of the dam (y)');`

`>>> hold on;`

`>>> plot(X, X_b*theta, '--', 'LineWidth', 2);`

`hold off;`

![Plot](figures/figure2.png)

Above plot shows the best fit line. The best fit line tells us that the model is not a good fit to the data because the data has a non-linear pattern. While visualizing the best fit as shown is one possible way to debug your learning algorithm, it is not always easy to visualize the data and model. In the next section, we will implement a function to generate learning curves that can help you debug your learning algorithm even if it is not easy to visualize the data

# Learning Curve for Linear Regression

We will generates the train and cross validation set errors needed to plot a learning curve. Call `learningCurve` function which will compute the train and cross-validation errors for dataset sizes from `1` up to `m`

For the cross-validation error, function will evaluate on the entire cross validation set (Xval and yval).
While calling cost function `linearRegCostFunction`  to compute the training and cross validation error, we will call the function with the lambda argument set to `0`. However, we still need to use lambda when running the training to obtain the theta parameters.


`>>> lambda = 0;`

### Add 1 as first column to matrix 'X' for x<sub>o</sub> = 1 

`>>> X_b = [ones(m, 1) X];`

`>>> Xval_b = [ones(size(Xval, 1), 1) Xval];`

### Call `learningCurve` function 

`>>> [error_train, error_val] = learningCurve(X_b, y, Xval_b, yval, lambda);`

Expected values of `error_train`: 
   `0.00000
    0.00000
    3.28660
    2.84268
   13.15405
   19.44396
   20.09852
   18.17286
   22.60941
   23.26146
   24.31725
   22.37391`

Expected values of `error_val`: 
  `205.121
   110.300
    45.010
    48.369
    35.865
    33.830
    31.971
    30.862
    31.136
    28.936
    29.551
    29.434`


## Plot curve

`>>> plot(1:m, error_train, 1:m, error_val);`

`>>> title('Learning curve for linear regression');`

`>>> legend('Train', 'Cross Validation');`

`>>> xlabel('Number of training examples');`

`>>> ylabel('Error');`

`>>> axis([0 13 0 150]);` 


![Plot](figures/figure3.png)


Since the model is underfitting the data, we expect graph will show "high bias".


# Feature Mapping for Polynomial Regression


The problem with our linear model was that it was too simple for the data and resulted in underfitting (high bias). We will address this problem by adding more features.

One solution to high bias problem is to use polynomial regression. We will call function `polyFeatures` to map each example into its powers.

### Degree of ploynomial:

`>>> p = 8;`

### Map X onto Polynomial Features and Normalize

`>>> X_poly = polyFeatures(X, p);`

### Feature Normalize

`>>> [X_poly, mu, sigma] = featureNormalize(X_poly);`  

### Add 1 as first column to matrix 'X' for x<sub>o</sub> = 1 

`>>> X_poly = [ones(m, 1), X_poly];`

### Map X_poly_test and normalize (using mu and sigma)

`>>> X_poly_test = polyFeatures(Xtest, p);`

`>>> X_poly_test = bsxfun(@minus, X_poly_test, mu);`

`>>> X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);`

### Add 1 as first column to matrix 'X' for x<sub>o</sub> = 1 
`>>> X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];`


### Map X_poly_val and normalize (using mu and sigma)

`>>> X_poly_val = polyFeatures(Xval, p);`

`>>> X_poly_val = bsxfun(@minus, X_poly_val, mu);`

`>>> X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);`

### Add 1 as first column to matrix 'X' for x<sub>o</sub> = 1 

`>>> X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];`


`>>> fprintf('Normalized Training Example 1:\n');`

`>>> fprintf('  %f  \n', X_poly(1, :));`

Expected values `X_poly`: 
  `1.000000  
  -0.362141  
  -0.755087  
  0.182226  
  -0.706190  
  0.306618  
  -0.590878  
  0.344516  
  -0.508481`

# Learning Curve for Polynomial Regression 

We will now experiment with polynomial regression with multiple values of lambda. The code below runs polynomial regression with &lambda;=`0`. 

We can try to run the code with different values of &lambda; to see how the fit and learning curve change.

## Try with &lambda;=`0` - No Regularization

`>>> lambda = 0;`

`>>> [theta, error_train, error_val] =  learningCurveForPolynomial(X_poly, X_poly_val, yval, X, y, lambda, mu, sigma, theta, p, m)`


![Plot](figures/figure4.png)

![Plot](figures/figure5.png)

Expected value of &theta;: 
  `11.2223
   11.3663
   14.6119
   11.1440
  -11.7166
  -14.4936
    6.7821
    3.0882
   -4.8796`


Expected value of `error_train`:
  `0.00000
   0.00000
   0.00000
   0.00000
   0.00000
   0.00013
   0.03425
   0.06811
   0.16175
   0.14154
   0.09077
   0.16925`


Expected value of `error_val`:
  `160.7219
   160.1215
    61.7548
    61.9289
     6.6147
    10.2985
     9.1291
    12.0046
     7.2088
     8.8345
     7.3669
    20.7642`

## Try with &lambda;=`1`

`>>> lambda = 1;`

`>>> [theta, error_train, error_val] =  learningCurveForPolynomial(X_poly, X_poly_val, yval, X, y, lambda, mu, sigma, theta, p, m)`


![Plot](figures/figure6.png)

![Plot](figures/figure7.png)

Expected value of &theta;:
  `11.21759
    8.58518
    5.21269
    3.59111
    2.28295
    1.76030
    1.06813
    1.10763
    0.39399`

Expected value of `error_train`:
  `0.00000
   0.04577
   2.91137
   1.49941
   1.19918
   0.92451
   1.54090
   1.42297
   1.55373
   1.44156
   1.30815
   2.07619`

Expected value of `error_val`:
  `138.8468
   143.5229
     5.5347
     6.6099
     6.6341
     8.2920
     5.7372
     5.5164
     6.2397
     5.9830
     6.0438
     4.2606`


## Try with &lambda;=`100`

`>>> lambda = 100;`

`>>> [theta, error_train, error_val] =  learningCurveForPolynomial(X_poly, X_poly_val, yval, X, y, lambda, mu, sigma, theta, p, m)`


![Plot](figures/figure8.png)

![Plot](figures/figure9.png)

Expected value of &theta;:
  `11.217589
    0.964209
    0.322966
    0.747428
    0.144230
    0.555808
    0.020283
    0.418693
   -0.058004`

Expected value of `error_train`:
  `9.8608e-32
   1.1411e-01
   1.0696e+02
   1.2174e+02
   1.0295e+02
   9.7170e+01
   8.3327e+01
   7.6492e+01
   7.1297e+01
   6.4351e+01
   5.8998e+01
   5.7977e+01`

Expected value of `error_val`:
  `138.847
   144.125
    70.863
    78.373
    63.845
    59.533
    59.585
    58.700
    59.564
    59.731
    60.410
    57.842`

## Conslustion 

For &lambda;=`1`, you should see that the polynomial fit is able to follow the datapoints very well,thus, obtaining a low training error. However, the polynomial fit is very complex and even drops off at the extremes. This is an indicator that the polynomial regression model is overfitting the training data and will not generalize well.

To better understand the problems with the unregularized (&lambda; = 0) model, you can see that the learning curve shows the same effect where the low training error is low, but the cross validation error is high. There is a gap between the training and cross validation errors, indicating a high variance problem.

For &lambda; = 1, you should see a polynomial fit that follows the data trend well and a learning curve showing that both the cross validation and training error converge to a relatively low value. This shows the &lambda; = 1 regularized polynomial regression model does not have the high bias or high-variance problems. In effect, it achieves a good trade-off between bias and variance.

For &lambda; = 100, you should see a polynomial fit that does not follow the data well. In this case, there is too much regularization and the model is unable to fit the training data.

By plotting and visually examining data, it show &lambda; = 1 be the best value. 

#  Validation for Selecting Lambda

In this section, we will implement an automated method to select the &lambda; parameter. Concretely, we will use a cross validation set to evaluate how good each &lambda; value is. After selecting the best &lambda; value using the cross validation set, we can then evaluate the model on the test set to estimate how well the model will perform on actual unseen data.

We will now implement validationCurve to test various values of lambda on a validation set. We will then use this to select the "best" lambda value.

We will try &lambda; in the following range: {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}.


`>>> [lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval);`

`>>> close all;`

`>>> plot(lambda_vec, error_train, lambda_vec, error_val);`

`>>> legend('Train', 'Cross Validation');`

`>>> xlabel('lambda');`

`>>> ylabel('Error');`

![Plot](figures/figure10.png)


Expected value of `error_train`:
   `0.10456
    0.15781
    0.18543
    0.22238
    0.28184
    0.45932
    0.92176
    2.07619
    4.90135
   16.09221`

Expected value of `error_val`:
   `7.4788
   17.2426
   18.9163
   17.0788
   12.8300
    7.5870
    4.6368
    4.2606
    3.8229
    9.9455`

## Conclusion

In this figure, we can see that the best value of &lambda; is around 3. Due to randomness in the training and validation splits of the dataset, the cross validation error can sometimes be lower than the training error.
