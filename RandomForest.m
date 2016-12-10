% Basic Random Forest on the word counts

%% Add paths
addpath(genpath('Models/'));
addpath(genpath('Utils/'));

%% Load data
load('train_set/words_train.mat');

%% Preprocess X
N = size(X, 1);
Xnew = full(X);
Xnew = dim_reduce(Xnew, 500);

%% Cross-Validation
[train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
    getYHatRandomForest(X_train, Y_train, X_test), Xnew, Y, 10);

fprintf('Train error: %f\n', train_error);
fprintf('Validation error: %f\n', val_error);
