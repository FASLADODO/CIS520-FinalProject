function [ X_projected, coeffs ] = getSemiSupervisedProjections()

%% Load data
labeled = load('train_set/words_train.mat');
unlabeled = load('train_set_unlabeled/words_train_unlabeled.mat');
X_labeled = labeled.X;
Y = labeled.Y;
X_unlabeled = unlabeled.X;
X_all = [X_labeled; X_unlabeled];
[n, p] = size(X_labeled);

%% Project Data Down
K = 500;
[X_all_projected, coeffs] = dim_reduce(full(X_all), K);
X_projected = X_all_projected(1:n, :);

end
