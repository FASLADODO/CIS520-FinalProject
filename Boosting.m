load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color.mat');

N = size(X, 1);

Xnew = [full(X), train_color];

t = templateTree('MaxNumSplits', 5);
ClassTreeEns = fitensemble(Xnew,Y,'AdaBoostM1',200,t,'Holdout', 0.5);

kflc = kfoldLoss(ClassTreeEns,'Mode','cumulative');
figure;
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');