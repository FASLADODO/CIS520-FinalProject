function [ yhat_train, yhat_test ] = getYHatNB( X_train, Y_train, X_test )
%GETYHATNB Summary of this function goes here
%   Detailed explanation goes here

Mdl = fitcnb(X_train, Y_train);
yhat_train = predict(Mdl, X_train);
yhat_test = predict(Mdl, X_test);

end

