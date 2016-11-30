load('glove_vecs200.mat');
load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color.mat');

N = size(X, 1);

Xnew = full(X); %, train_color];
indices = crossvalind('Kfold', N, 2);
train = indices == 1;
numTrain = sum(train);
test = ~train;
numTest = sum(test);
% Calculate training error
y_hat = evaluateGloVe(Xnew(train, :), vecs, happy, sad);
train_error = sum(abs(y_hat - Y(train))) / numTrain;

% Calculate testing error
y_hat = evaluateGloVe(Xnew(test, :), vecs, happy, sad);
test_error = sum(abs(y_hat - Y(test))) / numTest;

