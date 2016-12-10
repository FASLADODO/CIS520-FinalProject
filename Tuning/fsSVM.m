function [ test_error ] = fsSVM( X_train, Y_train, X_test, Y_test )
%FSSVM Summary of this function goes here
%   Detailed explanation goes here

[~, yhat_test] = getYHatSVM(X_train, Y_train, X_test);

numTest = length(Y_test);
test_error = sum(abs(yhat_test - Y_test)) / numTest;

end

