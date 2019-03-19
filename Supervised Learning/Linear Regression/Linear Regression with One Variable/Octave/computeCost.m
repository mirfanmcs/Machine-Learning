function J = computeCost(X, y, theta)

% Calculate cost function J of theta J(theta) using theta as the
%   parameter for linear regression to fit the data points in X and y

% Before calling function, append matrix X with 1 for x0=1
% Vector y is the known value/label of training data
   
% Algorithm:
% J(theta_0,theta_1) = (1/2*m) Sum [i=1 to m] (h(x)-y)^2 
%     where h(x)=theta_0 + theta_1*x
%     another way to calculate h(x) is using the matrix. h(x)= x * theta 
%            where theta is 2x1 vector  or transponse of 1x1 matrix 

% Vectorization implementation which is more efficient 

m = size(X,1); % number of training examples 
J = 0;  % initialize J to zero

h_x = X * theta; % computing h(x) 
sqrErrors = (h_x-y).^2;   % compute  (h(x)-y)^2. Using '.^' to compute squre on every item of matrix 

J = 1/(2*m) * sum(sqrErrors);   % calculate remaining part of formula (1/2*m) Sum [i=1 to m] (h(x)-y)^2  to the get the computed value of J(theta)

end
