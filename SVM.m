load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

N = size(X, 1);
%%  Hyper-parameter tuning with Cross-Validation
for numComponents = 100:100:1500
    disp(numComponents);
    Xnew = full(X(:, 1:1000));
    % Xnew = dim_reduce(Xnew, numComponents);

    [train_errors, test_errors] = crossValidate(@trainSVM, Xnew, Y, 1);

    disp(mean(train_errors));
    disp(mean(test_errors));
%     plot(1:10, train_errors, 'rx', 1:10, test_errors, 'b+');
end

%% Model Generation
% Xnew = full(X);
% [Xnew, coeffs] = dim_reduce(Xnew, 500);
% Mdl = fitcsvm(Xnew, Y);
% save('SVM_Model.mat', 'Mdl', 'coeffs');
