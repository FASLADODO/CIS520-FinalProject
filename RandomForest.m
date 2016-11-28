%% Basic Random Forest on the word counts
load('train_set/words_train.mat');
B = TreeBagger(12, full(X), Y);

%% Test locally on training data
yhat = str2num(cell2mat(B.predict(full(X))));