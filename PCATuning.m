% Script to find optimal PCA / SVM parameters

%% Initialization

% load data
load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

% setup variables
dims = 100:100:2000;
boxes = linspace(0.1, 6, 20);

train_errors = zeros(length(dims), length(boxes));
val_errors = zeros(length(dims), length(boxes));


%% Run trials
for d = 1:length(dims)
    % call PCA
    X = dim_reduce(full(X), dims(d));
    for b = 1:length(boxes)
        % get SVM crossval error
        [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
            getYHatSVM(X_train, Y_train, X_test, 'linear', 1), X, Y, 10);
        train_errors(d, b) = err1;
        val_errors(d, b) = err2;

        % print progress
        fprintf('PCA with %d dimensions and box size %d had %f training and %f val error\n',...
            dims(d), boxes(b), err1, err2);
    end
end

%% Plot results