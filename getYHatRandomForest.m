function [ yhat_train, yhat_test ] = getYHatRandomForest( X_train, Y_train, X_test, numTrees)
%getYHatSVM Trains a Random Forest on the data (X_train, Y_train) and generates y_hats
%   yhats(:, 1) = predictions of model on training data
%   yhats(:, 2) = predictions of model on testing data

if nargin < 4
    numTrees = 128;
end

Mdl = TreeBagger(numTrees, X_train, Y_train);
yhat_train = predict(Mdl, X_train);
yhat_test = predict(Mdl, X_test);

end

