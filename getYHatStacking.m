function [ yhat_train, yhat_test ] = getYHatStacking( X_train, Y_train, X_test, getYHatSVM, getYHatRandomForest, stackModel )
%TRAINSTACKING Summary of this function goes here
%   Detailed explanation goes here

if nargin < 6
    stackModel = 'Decision Tree';
end

numTrain = size(X_train, 1);
numTest = size(X_test, 1);
train_data = zeros(numTrain, 3);
test_data = zeros(numTest, 3);

% Build up train and test datasets
[train_data(:, 1), test_data(:, 1)] = getYHatSVM(X_train, Y_train, X_test, 'linear');
[train_data(:, 2), test_data(:, 2)] = getYHatSVM(X_train, Y_train, X_test, 'rbf');
[train_data(:, 3), test_data(:, 3)] = getYHatRandomForest(X_train, Y_train, X_test);

if isequal(stackModel, 'Decision Tree')
    % Train decision tree
    Mdl = fitctree(train_data, Y_train);
    yhat_train = predict(Mdl, train_data);
    yhat_test = predict(Mdl, test_data);
elseif isequal(stackModel, 'Naive Bayes')
    % Train Naive Bayes
    Mdl = fitcnb(train_data, Y_train);
    yhat_train = predict(Mdl, train_data);
    yhat_test = predict(Mdl, test_data);
elseif isequal(stackModel, 'Logistic Regression')
    % Train Logistic Regression
    B = mnrfit(train_data, Y_train + 1);
    % Haven't figured out how this one works yet
elseif isequal(stackModel, 'SVM')
    % Train SVM
    [yhat_train, yhat_test] = getYHatSVM(train_data, Y_train, test_data);
elseif isequal(stackModel, 'KNN')
    % Train KNN
    Mdl = fitcknn(train_data, Y_train);
    yhat_train = predict(Mdl, train_data);
    yhat_test = predict(Mdl, test_data);
end
end

