load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color.mat');

[n, p] = size(X);

Xnew = full(X); %[full(X)];
Xnew = dim_reduce(Xnew);
indices = crossvalind('Kfold', n, 2);
train = indices == 1;
numTrain = sum(train);
test = ~train;
numTest = sum(test);

Mdl = fitcnb(Xnew(train, :), Y(train));
y_hat = Mdl.predict(Xnew(train, :));
train_error = sum(abs(y_hat - Y(train))) / numTrain;

% Calculate testing error
y_hat = Mdl.predict(Xnew(test, :));
test_error = sum(abs(y_hat - Y(test))) / numTest;

fprintf('Train error: %d\n', train_error)
fprintf('Test error: %d\n', test_error)