function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Algorithm:
% J(theta) = (-1/m) Sum [i=1 to m] (y*log(h(x))+ (1-y)*log(1-h(x))) 
%    where h(x) = 1 / 1 + e ^ -theta*x
% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

z = X * theta;

% Computing h(x) 
h_x = sigmoid(z);   % h(x) = 1 / 1 + e ^ -theta*x

J = (-1 / m) * sum(y.*log(h_x) + (1 - y).*log(1 - h_x));

% Compute the partial derivatives and set grad to the partial
%   derivatives of the cost w.r.t. each parameter in theta

temp = sigmoid (z);
error = temp - y;
grad = (1 / m) * (X' * error);

end
