%% Add paths
addpath(genpath('Models/'));
addpath(genpath('Utils/'));

%% Load data
load('train_set/words_train.mat');
load('train_set/train_raw_img.mat');

%% Prepare image data by converting into grayscale vectors
% IM = arrayfun(@(x) imhist(rgb2gray(reshape_img(train_img(x, :)))), 1:4500, 'UniformOutput', false);
% IMres = cell2mat(IM)';

%% Preprocess X
Xnew = full(X);
Xnew = dim_reduce(Xnew, 500);

%% Cross-Validation
[train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
    getYHatKNN(X_train, Y_train, X_test, 10), Xnew, Y, 10);

fprintf('Train error: %f\n', train_error);
fprintf('Validation error: %f\n', val_error);