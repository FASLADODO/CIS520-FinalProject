%% Setup
% Constants
NUM_TREES = 128;
TRAIN_PROPORTION = 0.9;
K = 500;

% Load data
load('train_set/words_train.mat');
rng('default'); % For reproducibility
[n , p] = size(X);

% Get train/test splits
[train, val] = getTrainValSplits(n);

% Specify hyperparameters to be tuned
minLeafSizes = [1, 5, 10, 15, 20];
numPredictors = 2 .^ (1:floor(log2(K - 1) + 1));
reducedDimensions1 = 2 .^ (1:floor(log2(min(n, p) - 1)));
reducedDimensions2 = 100:200:900;

% Prepare X
X = full(X);

%% Tune minLeafSize hyperparameter individually
% Confirms that the best default minLeafSize is 1 (without pca), which is 
% TreeBagger's default for classification
minLeafSizeTrainError = zeros(1, length(minLeafSizes));
minLeafSizeValError = zeros(1, length(minLeafSizes));
for i = 1:length(minLeafSizes)
    fprintf('Training Random Forest with min leaf size: %d ...\n', ...
        minLeafSizes(i))
    randomForest = TreeBagger(NUM_TREES, X(train, :), Y(train, :), ...
        'MinLeafSize', minLeafSizes(i));
    [train_error, val_error] = evaluateModel(randomForest, X, Y, ...
        train, val);
    minLeafSizeTrainError(i) = train_error;
    minLeafSizeValError(i) = val_error;
end
plotTrainValError(minLeafSizes, minLeafSizeTrainError, minLeafSizeValError, ...
    'Random Forest Error for Various Min Leaf Sizes', 'Min Leaf Size')

%% Tune numPredictors hyperparameter individually
numPredictorsTrainError = zeros(1, length(numPredictors));
numPredictorsValError = zeros(1, length(numPredictors));
[X_projected, ~] = dim_reduce(X, K);
for i = 1:length(numPredictors)
    fprintf('Training Random Forest with number of predictors: %d ...\n', ...
        numPredictors(i))
    randomForest = TreeBagger(NUM_TREES, X_projected(train, :), Y(train, :), ...
        'NumPredictorstoSample', numPredictors(i));
    [train_error, val_error] = evaluateModel(randomForest, X_projected, Y, ...
        train, val);
    numPredictorsTrainError(i) = train_error;
    numPredictorsValError(i) = val_error;
end
plotTrainValError(numPredictors, numPredictorsTrainError, numPredictorsValError, ...
    'Random Forest Error for Various Numbers of Predictors', 'Numbers of Predictors')

%% Tune both together
minValError = Inf;
bestMinLeafSize = 1;
bestNumPredictors = 1;
for i = 1:length(minLeafSizes)
    for j = 1:length(numPredictors)
        randomForest = TreeBagger(NUM_TREES, X(train, :), Y(train, :), ...
            'MinLeafSize', minLeafSizes(i), ...
            'NumPredictorstoSample', numPredictors(j));
        [train_error, val_error] = evaluateModel(randomForest, X, Y, ...
            train, val);
        if val_error < minValError
            minValError = val_error;
            bestMinLeafSize = minLeafSizes(i);
            bestNumPredictors = numPredictors(j);
        end
    end
end

%% Tune PCA dimension hyperparameter individually
pcaTrainError = zeros(1, length(reducedDimensions1));
pcaValError = zeros(1, length(reducedDimensions1));
[~, coeffs] = dim_reduce(X);
for i = 1:length(reducedDimensions1)
    fprintf('Training Random Forest with PCA dimensionality reduction k: %d ...\n', ...
        reducedDimensions1(i))
    X_projected = X * coeffs(:, 1:reducedDimensions1(i));
    randomForest = TreeBagger(NUM_TREES, X_projected(train, :), Y(train, :));
    [train_error, val_error] = evaluateModel(randomForest, X_projected, Y, ...
        train, val);
    pcaTrainError(i) = train_error;
    pcaValError(i) = val_error;
end
plotTrainValError(reducedDimensions1, pcaTrainError, pcaValError, ...
    'Random Forest Error for Various PCA dimensions', 'k')

%% Tune PCA dimension hyperparameter individually at at finer scale
pcaTrainError = zeros(1, length(reducedDimensions2));
pcaValError = zeros(1, length(reducedDimensions2));
[~, coeffs] = dim_reduce(X);
for i = 1:length(reducedDimensions2)
    fprintf('Training Random Forest with PCA dimensionality reduction k: %d ...\n', ...
        reducedDimensions2(i))
    X_projected = X * coeffs(:, 1:reducedDimensions2(i));
    randomForest = TreeBagger(NUM_TREES, X_projected(train, :), Y(train, :));
    [train_error, val_error] = evaluateModel(randomForest, X_projected, Y, ...
        train, val);
    pcaTrainError(i) = train_error;
    pcaValError(i) = val_error;
end
plotTrainValError(reducedDimensions2, pcaTrainError, pcaValError, ...
    'Random Forest Error for Various PCA dimensions', 'k')

%% Train best Random Forest
[~, coeffs] = dim_reduce(X);
X_projected = X * coeffs(:, 1:256);
randomForest = TreeBagger(NUM_TREES, X_projected(train, :), Y(train, :));
[train_error, val_error] = evaluateModel(randomForest, X_projected, Y, ...
    train, val);
