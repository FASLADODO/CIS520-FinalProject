load('train_set/words_train.mat');

X = full(X);
N = size(X, 1);

%%  Hyper-parameter tuning with Cross-Validation
% for numComponents = 100:100:1500
%     fprintf('Number of Components: %d\n', numComponents);
    Xnew = dim_reduce(X, 500);

    [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatEnsemble(X_train, Y_train, X_test), Xnew, Y, 1);

    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', val_error);
% end
