function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% Gradient descent is used to minimize cost function J
% Vectorisation implementation. Applicable for any no for features. 
% Theta should be a nx1 vector where n = No of feature + 1  

%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Algorithm
% theta_i= theta_i - alpha * Sum [i=1 to m] (h(x)-y)*x_i 
%    where i= training data 


m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    z = X * theta;

    % Computing h(x) 
    h_x = sigmoid(z);   % h(x) = 1 / 1 + e ^ -theta*x

    sqrErrors = h_x - y;
    newX = sqrErrors' * X;
    theta = theta - ((alpha) * newX');

    costFunction(theta, X, y)
   
    % Save the cost J in every iteration    
    [J_history(iter), grad] = costFunction(theta, X, y);

end

end
