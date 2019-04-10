function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i


% We will use Advanced Optimization custom function fmincg instead of fminunc.
% fmincg works similar to fminunc, but is more efficient for dealing with a large number of parameters.

% In Octave/MATLAB, evaluating the expression a == b for a vector a (of size m×1) and scalar b will return 
% a vector of the same size as a with ones at positions where the elements of a are equal to b and zeroes 
% where they are different. 
% For example: 
%   a = 1:10;
%   b = 3;
%   a == b    
% last statement will return '0  0  1  0  0  0  0  0  0  0'

% We will use this feature to select one vs all for a selected value. 
% For example y==2 will return value of y with 1 set for position of alls 2's and rest will be set to 0.   


%Initialization
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix for x0=1
X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels
	[theta] = fmincg (@(t)(costFunctionReg(t, X, (y == c), lambda)), initial_theta, options);
	%[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, (y == c), lambda)), initial_theta, options);
    all_theta(c,:) = theta';
endfor

end
