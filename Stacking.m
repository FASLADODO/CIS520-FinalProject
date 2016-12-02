load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

N = size(X, 1);
Xnew = dim_reduce(full(X), 500);

[train_error, test_error] = crossValError(@(X_train, Y_train, X_test) ...
    getYHatStacking(X_train, Y_train, X_test, @getYHatSVM, @getYHatRandomForest, 'KNN'), ...
    Xnew, Y, 1);

disp(train_error);
disp(test_error);