function [ train_errors, val_errors ] = crossValidate( getYhats, X, Y, numFolds )
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
val_errors = ones(numFolds, 1);
for i = 1:numFolds
    if numFolds > 1
        fprintf('Current Fold: %d\n', i);
        val = (indices == i); 
        train = ~val;
    else
        % We aren't doing cross-validation at this point,
        % so train on 70% of the data and test on the other
        % 30%
        [train, val] = getTrainValSplits(N, 0.7);
    end
    
    numTrain = sum(train);
    numTest = sum(val);

    [yhat_train, yhat_val] = getYhats(X(train, :), Y(train), X(val, :));
    
    fprintf('Confusion matrix: %s\n', mat2str(generateConfusionMatrix(Y(val), yhat_val)));

    train_errors(i) = full(sum(abs(yhat_train - Y(train))) / numTrain);
    val_errors(i) = full(sum(abs(yhat_val - Y(val))) / numTest);
end
end
