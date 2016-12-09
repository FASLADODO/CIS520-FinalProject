function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 0 for sad)

% %% Load data
labeled = load('words_train.mat');
unlabeled = load('words_train_unlabeled.mat');
load('coeffs.mat');
X_labeled = labeled.X;
Y = labeled.Y;
X_unlabeled = unlabeled.X;
X_all = [X_labeled; X_unlabeled];
[n, p] = size(X_labeled);

%% Project data down
K = 500;
X_all_projected = X_all * coeffs;
X_projected = X_all_projected(1:n, :);

%% Train model
Mdl = fitcsvm(X_projected, Y, 'KernelFunction', 'linear', 'BoxConstraint', 0.5);
% load('SVM_Model.mat');
% Xnew = word_counts * coeffs(:, 1:K);
Xnew = word_counts * coeffs;
Y_hat = full(predict(Mdl, full(Xnew)));

end
