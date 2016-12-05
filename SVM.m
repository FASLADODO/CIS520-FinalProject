load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

N = size(X, 1);
X = dim_reduce(full(X), 500);

%% Feature Selection with sequentialfs
% opts = statset('display','iter');
% [fs, history] = sequentialfs(@fsSVM, X, Y, 'options', opts);

%%  Hyper-parameter tuning with Cross-Validation
% [~, coeffs] = dim_reduce(X);
% for boxConstraint = 1:10
%     disp(boxConstraint);
%     Xnew = full(X*coeffs(:, 1:numComponents));

    [train_error, test_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatSVM(X_train, Y_train, X_test, 'linear', boxConstraint), X, Y, 3);

    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', test_error);
%     plot(1:10, train_errors, 'rx', 1:10, test_errors, 'b+');
% end

%% Model Generation
% [Xnew, coeffs] = dim_reduce(Xnew, 500);
% Mdl = fitcsvm(Xnew, Y);
% save('SVM_Model.mat', 'Mdl', 'coeffs');