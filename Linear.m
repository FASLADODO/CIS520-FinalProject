%% Load data
load('train_set/words_train.mat');

n = size(X, 1);
X_projected = dim_reduce(full(X), 500);

for lambda = linspace(.0001, .001, 20)
    fprintf('Lambda = %f\n', lambda);
    %% Linear Shits
    [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatLinear(X_train, Y_train, X_test, lambda), ...
        X_projected, Y, 10);

    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', val_error);
end