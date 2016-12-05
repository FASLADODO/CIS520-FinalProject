load('train_set/words_train.mat');
load('train_set/train_raw_img.mat');

% prepare image data by converting into grayscale vectors
IM = arrayfun(@(x) imhist(rgb2gray(reshape_img(train_img(x, :)))), 1:4500, 'UniformOutput', false);
IMres = cell2mat(IM)';

% train model
nn = 50;
train_errors = zeros(nn, 1);
val_errors = zeros(nn, 1);
for i = 1:nn
    [train_errors(i), val_errors(i)] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatKNN(X_train, Y_train, X_test, i), IMres, Y, 3);
    fprintf('Training and val errors for %d nn are %f and %f\n', ...
        i, train_errors(i), val_errors(i));
end