function [ train_errors, test_errors ] = crossValidate( getYhats, X, Y, numFolds )
%CROSSVALIDATE Summary of this function goes here
%   getYhats - function that takes in X(train, :), Y(train), and X(test, :)
%   and produces [predictions for X(train, :), pred. for X(test, :)]
%   k - number of folds

N = size(X, 1);

indices = crossvalind('Kfold', N, numFolds);
train_errors = ones(numFolds, 1);
test_errors = ones(numFolds, 1);
for i = 1:numFolds
    if numFolds > 1
        test = (indices == i); 
        train = ~test;
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

