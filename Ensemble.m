labeled = load('train_set/words_train.mat');
unlabeled = load('train_set_unlabeled/words_train_unlabeled.mat');
X_labeled = labeled.X;
Y = labeled.Y;
X_unlabeled = unlabeled.X;
X_all = [X_labeled; X_unlabeled];
[n, p] = size(X_labeled);

%% Project data down
K = 500;
X_all_projected = dim_reduce(full(X_all), K);
X_projected = X_all_projected(1:n, :);

X = full(X(1:n, :));

%%  Hyper-parameter tuning with Cross-Validation
for numComponents = 1387 % 200:100:1500
    fprintf('Number of Components: %d\n', numComponents);
    Xnew = dim_reduce(X, numComponents);

    [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatEnsemble(X_train, Y_train, X_test), Xnew, Y, 1);

    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', val_error);
end
