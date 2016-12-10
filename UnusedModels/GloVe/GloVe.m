%% Add paths
addpath(genpath('Models/'));
addpath(genpath('Utils/'));

%% Load data
load('glove_vecs200.mat');
load('train_set/words_train.mat');

%% Preprocess X
N = size(X, 1);
X = full(X);

%% Cross-validate
Xnew = gloveTransform(X, vecs);
% Xnew = dim_reduce(X, numComponents);

[train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
    getYHatSVM(X_train, Y_train, X_test, 'rbf', 1), Xnew, Y, 10);

fprintf('Train error: %f\n', train_error);
fprintf('Validation error: %f\n', val_error);