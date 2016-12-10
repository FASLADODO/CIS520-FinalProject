%% Load data
load('train_set/words_train.mat');

%% Preprocess X
Xnew = full(X);
Xnew = dim_reduce(Xnew, 500);

%% Cross-Validation
[train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
    getYHatSVM(X_train, Y_train, X_test, 'linear', 1), Xnew, Y, 10);

fprintf('Train error: %f\n', train_error);
fprintf('Validation error: %f\n', val_error);