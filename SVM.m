load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

N = size(X, 1);
%%  Hyper-parameter tuning with Cross-Validation
Xnew = full(X);
[~, coeffs] = dim_reduce(Xnew);
for numComponents = 100:100:1500
    disp(numComponents);
    Xnew = full(X*coeffs(:, 1:numComponents));

    [train_errors, test_errors] = crossValidate(@(X_train, Y_train, X_test) ...
        getYHatSVM(X_train, Y_train, X_test, 'polynomial'), Xnew, Y, 3);

    disp(mean(train_errors));
    disp(mean(test_errors));
%     plot(1:10, train_errors, 'rx', 1:10, test_errors, 'b+');
end

%% Model Generation
% Xnew = full(X);
% [Xnew, coeffs] = dim_reduce(Xnew, 500);
% Mdl = fitcsvm(Xnew, Y);
% save('SVM_Model.mat', 'Mdl', 'coeffs');