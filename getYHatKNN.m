function [ yhat_train, yhat_test ] = getYHatKNN( X_train, Y_train, X_test, nn)
%getYHatKNN Trains a k-nn on the data (X_train, Y_train) and generates y_hats
%   yhats(:, 1) = predictions of model on training data
%   yhats(:, 2) = predictions of model on testing data

if nargin < 4
    nn = 5;
end

Mdl = fitcknn(X_train, Y_train, 'NumNeighbors', nn, 'Standardize', 1);
yhat_train = predict(Mdl, X_train);
yhat_test = predict(Mdl, X_test);
end