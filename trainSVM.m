function [ yhat_train, yhat_test ] = trainSVM( X_train, Y_train, X_test )
%TRAINSVM Trains an SVM on the data (X_train, Y_train) and generates y_hats
%   yhats(:, 1) = predictions of model on training data
%   yhats(:, 2) = predictions of model on testing data
    Mdl = fitcsvm(X_train, Y_train);
    yhat_train = predict(Mdl, X_train);
    yhat_test = predict(Mdl, X_test);
end

