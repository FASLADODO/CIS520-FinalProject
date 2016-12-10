function [ train_error, test_error ] = evaluateModel( Mdl, X, Y, train, test )
%EVALUATEMODEL Summary of this function goes here
%   Detailed explanation goes here

numTrain = sum(train);
numTest = sum(test);

% Calculate training error
y_hat = predict(Mdl, X(train, :));
if iscell(y_hat)
    % y_hat = str2double(y_hat);
    y_hat = cellfun(@str2double, y_hat); % apparently faster than the line above
end

train_error = full(sum(abs(y_hat - Y(train))) / numTrain);

% Calculate testing error
y_hat = predict(Mdl, X(test, :));
if iscell(y_hat)
    % y_hat = str2double(y_hat);
    y_hat = cellfun(@str2double, y_hat); % apparently faster than the line above
end

test_error = full(sum(abs(y_hat - Y(test))) / numTest);

end

