function [ yhat_train, yhat_test ] = getYHatGMM( X_train, ~, X_test, k )
%GETYHATGMM Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4
    k = 2;
end

Mdl = fitgmdist(X_train, k);
yhat_train = predict(Mdl, X_train);
yhat_test = predict(Mdl, X_test);

end

