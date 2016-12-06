function [ yhat_train, yhat_test ] = getYHatSVM( X_train, Y_train, X_test, KernelFunc, box)
%getYHatSVM Trains an SVM on the data (X_train, Y_train) and generates y_hats
%   yhats(:, 1) = predictions of model on training data
%   yhats(:, 2) = predictions of model on testing data

if nargin < 4
    KernelFunc = 'linear';
end

if nargin < 5
    box = 1;
end

%Mdl = fitcsvm(X_train, Y_train, 'OptimizeHyperparameters','auto',...
%    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%    'expected-improvement-plus'), 'CacheSize', 'maximal');
Mdl = fitcsvm(X_train, Y_train, 'KernelFunction', KernelFunc, 'BoxConstraint', box);
% Mdl = fitcdiscr(X_train, Y_train, 'discrimType', 'pseudoLinear');
yhat_train = predict(Mdl, X_train);
yhat_test = predict(Mdl, X_test);
end

