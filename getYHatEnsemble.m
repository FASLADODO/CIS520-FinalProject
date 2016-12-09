function [ yhat_train, yhat_test ] = getYHatEnsemble(X_train, Y_train, X_test)
%getYHatSVM Trains an SVM on the data (X_train, Y_train) and generates y_hats
%   yhats(:, 1) = predictions of model on training data
%   yhats(:, 2) = predictions of model on testing data

% Mdl = fitcensemble(X_train, Y_train, ...
%     'Method', 'Subspace', 'Learners', 'Discriminant', 'NPredToSample', 200);

Mdl = fitcensemble(X_train, Y_train, ...
    'Method', 'Subspace', 'Learners', 'Discriminant', ...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));

yhat_train = predict(Mdl, X_train);
yhat_test = predict(Mdl, X_test);
end
