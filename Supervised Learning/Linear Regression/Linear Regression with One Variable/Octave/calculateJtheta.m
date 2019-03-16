function [theta0_vals,theta1_vals,J_vals] = calculateJtheta(X, y)

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  theta = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, theta);
    end
end

end

