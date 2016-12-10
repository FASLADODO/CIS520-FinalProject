%% Add paths
addpath(genpath('Models/'));
addpath(genpath('Utils/'));

%% Load data
load('train_set/words_train.mat');

%% Preprocess X
N = size(X, 1);
Xnew = full(X);

%% Run model
t = templateTree('MaxNumSplits', 5);
ClassTreeEns = fitensemble(Xnew,Y,'AdaBoostM1',200,t,'Holdout', 0.5);

kflc = kfoldLoss(ClassTreeEns,'Mode','cumulative');
figure;
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');