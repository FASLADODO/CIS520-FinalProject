function [ train_errors, test_errors ] = crossValidate( getYhats, X, Y, numFolds )
%CROSSVALIDATE Summary of this function goes here
%   getYhats - function that takes in X(train, :), Y(train), and X(test, :)
%   and produces [predictions for X(train, :), pred. for X(test, :)]
%   numFolds - number of folds

if nargin < 4
    numFolds = 10;
end

N = size(X, 1);

indices = crossvalind('Kfold', N, numFolds);
train_errors = ones(numFolds, 1);
test_errors = ones(numFolds, 1);
for i = 1:numFolds
    fprintf('Current Fold: %d\n', i);
    if numFolds > 2
        test = (indices == i); 
        train = ~test;
    elseif numFolds == 2
        % We aren't doing cross-validation at this point,
        % so train on 90% of the data and test on the other
        % 10%
        [train, test] = getTrainValSplits(N, 0.9);
    else
        % numFolds is 1, so just train on entirety of X
        train = TRUE(N, 1);
        test = FALSE(N, 1);
    end
    
    numTrain = sum(train);
    numTest = sum(test);

    [yhat_train, yhat_test] = getYhats(X(train, :), Y(train), X(test, :));
    
    train_errors(i) = full(sum(abs(yhat_train - Y(train))) / numTrain);
    test_errors(i) = full(sum(abs(yhat_test - Y(test))) / numTest);
end
end
