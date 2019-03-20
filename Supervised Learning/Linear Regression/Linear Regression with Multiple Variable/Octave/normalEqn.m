function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

%Calculate minimum of theta using Normal Equation. This is another method to calculate minimum theta like Gradient Descent 

%Algorithm 
% theta = (X'*X)inv * X' * y

theta = pinv(X'*X) * X' * y;

end
