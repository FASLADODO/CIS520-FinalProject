%% Basic Random Forest on the word counts
load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color.mat');
% 
N = size(X, 1);
% indices = crossvalind('Kfold', N, 10);
% cp = classperf(Y);
% errorrate_sum = 0;
% for i = 1:10
%     test = (indices == i); 
%     train = ~test;
%     B = TreeBagger(64, full(X(train, :)), Y(train));
%     y_hat = str2num(cell2mat(B.predict(full(X(test, :)))));
%     classperf(cp, y_hat, test)
%     errorrate_sum = errorrate_sum + cp.ErrorRate;
% end
% 
% disp(errorrate_sum / 10);

% Hyperparameter tuning
Xnew = [full(X)]; %, train_color];
[Xnew, coeffs] = dim_reduce(Xnew, 800);
indices = crossvalind('Kfold', N, 2);
train = indices == 1;
numTrain = sum(train);
test = ~train;
numTest = sum(test);
test_errors = ones(6, 1);
train_errors = ones(6, 1);
for i = (1:6)
    disp(i);
    numTrees = 2 ^ i;
    B = TreeBagger(numTrees, Xnew(train, :), Y(train));
    % Calculate training error
%     y_hat = str2num(cell2mat(B.predict(Xnew(train, :))));
%     train_errors(i) = sum(abs(y_hat - Y(train))) / numTrain;
%     
%     % Calculate testing error
%     y_hat = str2num(cell2mat(B.predict(Xnew(test, :))));
%     test_errors(i) = sum(abs(y_hat - Y(test))) / numTest;
    [train_errors(i), test_errors(i)] = evaluateModel(B, Xnew, Y, train, test);
end        

numTrees = 2 .^ (1:6);
plot(numTrees, train_errors, 'rx', numTrees, test_errors, 'b+');
title('Hyperparameter tuning the number of trees for TreeBagger');
xlabel('Number of Trees');
ylabel('Error Rate');
legend('Training errors', 'Testing errors');

% B = TreeBagger(12, full(X), Y);

%% Test locally on training data
% yhat = str2num(cell2mat(B.predict(full(X))));
% error = sum(abs(yhat - Y));