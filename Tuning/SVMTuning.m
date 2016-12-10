%% Add paths
addpath(genpath('../'));
addpath(genpath('../Utils/'));

%% Load data
load('train_set/words_train.mat');

n = size(X, 1);
X_projected = dim_reduce(full(X), 500);

%% Linear Stuffs
[train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatLinear(X_train, Y_train, X_test), ...
        X_projected, Y, 10);

%% Broad C hyperparameter tuning
boxConstraints = 10 .^ (-2:2);
trainErrors = zeros(1, length(boxConstraints));
valErrors = zeros(1, length(boxConstraints));
for i = 1:length(boxConstraints)
    fprintf('Training SVM with C: %d ...\n', boxConstraints(i))
    [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatSVM(X_train, Y_train, X_test, 'linear', boxConstraints(i)), ...
        X_projected, Y, 10);
    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', val_error);
    trainErrors(i) = train_error;
    valErrors(i) = val_error;
    plot(boxConstraints(1:i), trainErrors(1:i), 'rx', ...
        boxConstraints(1:i), valErrors(1:i), 'b+');
end

%% More Specific C hyperparameter tuning
boxConstraints = [.05, .07, .1, .5, .7, 1, 1.5];
trainErrors = zeros(1, length(boxConstraints));
valErrors = zeros(1, length(boxConstraints));
for i = 1:length(boxConstraints)
    fprintf('Training SVM with C: %d ...\n', boxConstraints(i))
    [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatSVM(X_train, Y_train, X_test, 'linear', boxConstraints(i)), ...
        X_projected, Y, 10);
    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', val_error);
    trainErrors(i) = train_error;
    valErrors(i) = val_error;
    plot(boxConstraints(1:i), trainErrors(1:i), 'rx', ...
        boxConstraints(1:i), valErrors(1:i), 'b+');
end

%% Even More Specific C hyperparameter tuning
lowerBound = 0.3;
upperBound = 0.7;
boxConstraints = lowerBound + rand(1, 10) * (upperBound - lowerBound);
trainErrors = zeros(1, length(boxConstraints));
valErrors = zeros(1, length(boxConstraints));
for i = 1:length(boxConstraints)
    fprintf('Training SVM with C: %d ...\n', boxConstraints(i))
    [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatSVM(X_train, Y_train, X_test, 'linear', boxConstraints(i)), ...
        X_projected, Y, 10);
    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', val_error);
    trainErrors(i) = train_error;
    valErrors(i) = val_error;
end
plot(boxConstraints, trainErrors, 'rx', ...
    boxConstraints, valErrors, 'b+');
[minValError, minIndex] = min(valErrors);
fprintf('Best value for C: %d\nYielded Validation error: %d', ...
    boxConstraints(minIndex), minValError)

%% More Specific C hyperparameter tuning
boxConstraints = [1, 5, 10, 15, 20];
trainErrors = zeros(1, length(boxConstraints));
valErrors = zeros(1, length(boxConstraints));
for i = 1:length(boxConstraints)
    fprintf('Training SVM with C: %d ...\n', boxConstraints(i))
    [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatSVM(X_train, Y_train, X_test, 'linear', boxConstraints(i)), ...
        X_projected, Y, 10);
    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', val_error);
    trainErrors(i) = train_error;
    valErrors(i) = val_error;
end
plot(boxConstraints, trainErrors, 'rx', ...
    boxConstraints, valErrors, 'b+');
[minValError, minIndex] = min(valErrors);
fprintf('Best value for C: %d\nYielded Validation error: %d', ...
    boxConstraints(minIndex), minValError)

%% Tune both PCA and C at once
[~, coeffs] = dim_reduce(X);
boxConstraints = [.1, .5, 1, 5, 10];
pcaDimensions = [100, 200, 250, 300, 350, 400, 450, 500, 550, 600];
bestParams = [1, 500];
trainErrors = zeros(1, length(boxConstraints) * length(pcaDimensions));
valErrors = zeros(1, length(boxConstraints) * length(pcaDimensions));
bestValError = Inf;
for i = 1:length(boxConstraints)
    for j = 1:length(pcaDimensions)
        fprintf('Training SVM with C: %d and PCA: %d...\n', ...
            boxConstraints(i), pcaDimensions(j))
        X_projected = X * coeffs(:, 1:pcaDimensions(j));
        [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
            getYHatSVM(X_train, Y_train, X_test, 'linear', boxConstraints(i)), ...
            X_projected, Y, 10);
        fprintf('Train error: %f\n', train_error);
        fprintf('Validation error: %f\n', val_error);
        if val_error < bestValError
           bestValError = val_error;
           bestParams = [boxConstraints(i), pcaDimensions(j)];
        end
        trainErrors((i - 1) * length(boxConstraints) + j) = train_error;
        valErrors((i - 1) * length(boxConstraints) + j) = val_error;
    end
end
fprintf('Best value for C: %d and PCA: %d', ...
    bestParams(1), bestParams(2))

%% Tune both PCA and C at once (broad_svm_tune.mat)
[~, coeffs] = dim_reduce(X);
boxConstraints = [.1, .5, 1, 5];
pcaDimensions = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000];
bestParams = [1, 500];
trainErrors = zeros(length(boxConstraints), length(pcaDimensions));
valErrors = zeros(length(boxConstraints), length(pcaDimensions));
bestValError = Inf;
for i = 1:length(boxConstraints)
    for j = 1:length(pcaDimensions)
        fprintf('Training SVM with C: %d and PCA: %d...\n', ...
            boxConstraints(i), pcaDimensions(j))
        X_projected = X * coeffs(:, 1:pcaDimensions(j));
        [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
            getYHatSVM(X_train, Y_train, X_test, 'linear', boxConstraints(i)), ...
            X_projected, Y, 10);
        fprintf('Train error: %f\n', train_error);
        fprintf('Validation error: %f\n', val_error);
        if val_error < bestValError
           bestValError = val_error;
           bestParams = [boxConstraints(i), pcaDimensions(j)];
           fprintf('New high score, bitches! ... %d', bestValError)
        end
        trainErrors(i, j) = train_error;
        valErrors(i, j) = val_error;
    end
end
fprintf('Best value for C: %d and PCA: %d', ...
    bestParams(1), bestParams(2))

%% Tune both PCA and C at once (specific_svm_tune.mat)
[~, coeffs] = dim_reduce(X);
boxConstraints = [.001, .01, .1, .5];
pcaDimensions = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000];
bestParams = [1, 500];
trainErrors = zeros(length(boxConstraints), length(pcaDimensions));
valErrors = zeros(length(boxConstraints), length(pcaDimensions));
bestValError = Inf;
for i = 1:length(boxConstraints)
    for j = 1:length(pcaDimensions)
        fprintf('Training SVM with C: %d and PCA: %d...\n', ...
            boxConstraints(i), pcaDimensions(j))
        X_projected = X * coeffs(:, 1:pcaDimensions(j));
        [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
            getYHatSVM(X_train, Y_train, X_test, 'linear', boxConstraints(i)), ...
            X_projected, Y, 10);
        fprintf('Train error: %f\n', train_error);
        fprintf('Validation error: %f\n', val_error);
        if val_error < bestValError
           bestValError = val_error;
           bestParams = [boxConstraints(i), pcaDimensions(j)];
           fprintf('New high score, bitches! ... %d', bestValError)
        end
        trainErrors(i, j) = train_error;
        valErrors(i, j) = val_error;
    end
end
fprintf('Best value for C: %d and PCA: %d', ...
    bestParams(1), bestParams(2))

%%
[train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
            getYHatSVM(X_train, Y_train, X_test, 'linear', 1), ...
            X_projected, Y, 10);

%% Model Generation
% [Xnew, coeffs] = dim_reduce(Xnew, 500);
% Mdl = fitcsvm(Xnew, Y);
% save('SVM_Model.mat', 'Mdl', 'coeffs');