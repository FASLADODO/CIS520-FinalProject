%% Add paths
addpath(genpath('../'));
addpath(genpath('../Utils/'));

%% Load data
load('train_set/words_train.mat');

n = size(X, 1);
% X_projected = dim_reduce(full(X), 500);
X_projected = full(X);

Ynew = getOneHotY(Y);
net = patternnet(500);
net.layers{1}.transferFcn = 'poslin';
net.trainFcn = 'trainscg';

net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

net.performParam.regularization = 0;
net = train(net, X_projected', Ynew);
