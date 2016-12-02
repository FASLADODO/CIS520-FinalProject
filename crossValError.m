function [ train_error, test_error ] = crossValError( getYhats, X, Y, numFolds )
%crossValError Gets the cross-validation err
%   getYhats - function that takes in X(train, :), Y(train), and X(test, :)
%   and produces [predictions for X(train, :), pred. for X(test, :)]
%   numFolds - number of folds

[train_errors, test_errors] = crossValidate(getYhats, X, Y, numFolds);
train_error = mean(train_errors);
test_error = mean(test_errors);

end

