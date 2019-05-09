function  [theta, error_train, error_val] = learningCurveForPolynomial(X_poly, X_poly_val, yval, X, y, lambda, mu, sigma, p, m)

theta = trainLinearReg(X_poly, y, lambda);

figure(1);

plot(X, y, 'rx', 'MarkerSize', 5, 'LineWidth', 1.5);
    
plotFit(min(X), max(X), mu, sigma, theta, p);
    
xlabel('Change in water level (x)');
    
ylabel('Water flowing out of the dam (y)');
    
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
    
figure(2);
    
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);

plot(1:m, error_train, 1:m, error_val);
    
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
    
xlabel('Number of training examples');
    
ylabel('Error');
    
axis([0 13 0 100]);
    
legend('Train', 'Cross Validation');

% print(error_train);
% print(error_train);

end
