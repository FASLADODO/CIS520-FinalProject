function [ train_error, val_error ] = crossValError( getYhats, X, Y, numFolds )
%crossValError Gets the cross-validation err
%   getYhats - function that takes in X(train, :), Y(train), and X(test, :)
%   and produces [predictions for X(train, :), pred. for X(test, :)]
%   numFolds - number of folds

if nargin < 4
    numFolds = 10;
end

[train_errors, val_errors] = crossValidate(getYhats, X, Y, numFolds);
train_error = mean(train_errors);
val_error = mean(val_errors);

end

