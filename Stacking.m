%% Add paths
addpath(genpath('Models/'));
addpath(genpath('Utils/'));

%% Load data
load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

%% Preprocess X
N = size(X, 1);
X = dim_reduce(full(X), 500);

%% Cross-validate
[train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
    getYHatStacking(X_train, Y_train, X_test, @getYHatSVM, @getYHatRandomForest, 'Test'), ...
    X, Y, 10);

fprintf('Train error: %f\n', train_error);
fprintf('Validation error: %f\n', val_error);