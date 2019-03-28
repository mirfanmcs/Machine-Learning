function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
%figure;

% Find Indices of Positive and Negative Examples
pos = find(y==1);
neg = find(y == 0);

% Plot data
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 4);

hold on 

plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 4);

xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('Accepted', 'Rejected')

title('Scatter plot of training data')
hold off


end
