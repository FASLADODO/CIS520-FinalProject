function [ yhat_train, yhat_test ] = getYHatLinear(X_train, Y_train, X_test, lambda, learner)
%getYHatSVM Trains an SVM on the data (X_train, Y_train) and generates y_hats
%   yhats(:, 1) = predictions of model on training data
%   yhats(:, 2) = predictions of model on testing data

% Mdl = fitclinear(X_train, Y_train, ...
%     'Learner', 'logistic', 'Solver', 'sparsa', 'Regularization','lasso',...
%     'Lambda', 'auto','GradientTolerance', 1e-8);

% Mdl = fitclinear(X_train, Y_train, ...
%     'Learner', 'logistic', 'Solver', 'bfgs', 'Regularization', 'ridge',...
%     'Lambda', lambda);

% Mdl = fitclinear(X_train,Y_train,'Solver','sparsa',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus'));

if nargin < 5
    learner = 'svm';
end

Mdl = fitclinear(X_train, Y_train, ...
    'Learner', learner, 'Solver', 'sparsa', 'Regularization', 'lasso', ...
    'Lambda', lambda);

yhat_train = predict(Mdl, X_train);
yhat_test = predict(Mdl, X_test);
end
