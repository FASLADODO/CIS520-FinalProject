load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color.mat');

Xnew = full(X); %[full(X)];
Xnew = dim_reduce(Xnew);
indices = crossvalind('Kfold', N, 2);
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

disp(train_error);
disp(test_error);