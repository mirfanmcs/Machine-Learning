% Gradient descent is used to minimize cost function J
% Implementation for single variable/feature 

function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Algorithm
% theta_0= theta_0 - alpha * (1/m) Sum [i=1 to m] (h(x)-y)*x_0 
% theta_1= theta_1 - alpha * (1/m) Sum [i=1 to m] (h(x)-y)*x_1
%    where x_0=1 

% Vectorization implementation which is more efficient 

m = size(X,1); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    h_x = X * theta; % computing h(x) 
    sqrErrors = h_x-y;
    theta_0 = theta(1);
    theta_1 = theta(2);
    
    temp_0 = theta_0 - alpha * (1/m) * sum(sqrErrors .* X(:,1));
    temp_1 = theta_1 - alpha * (1/m) * sum(sqrErrors .* X(:,2));
 
    theta = [temp_0;temp_1];
      
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
