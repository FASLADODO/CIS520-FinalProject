%% Setup
% Constants
NUM_TREES = 128;
TRAIN_PROPORTION = 0.9;

% Load data
load('train_set/words_train.mat');
rng('default'); % For reproducibility
[n , p] = size(X);

% Get train/test splits
randPerm = randperm(n);
train = randPerm(1:floor(TRAIN_PROPORTION * n));
val = randPerm(floor(TRAIN_PROPORTION * n) + 1:end);

% Specify hyperparameters to be tuned
minLeafSizes = [1, 5, 10, 15, 20];
numPredictors = 2 .^ (1:floor(log2(p - 1)));

% Prepare X
X = full(X);

%% Tune minLeafSize hyperparameter individually
% Confirms that the best default minLeafSize is 1, which is TreeBagger's
% default for classification
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
for i = 1:length(numPredictors)
    randomForest = TreeBagger(NUM_TREES, X(train, :), Y(train, :), ...
        'NumPredictorstoSample', numPredictors(i));
    [train_error, val_error] = evaluateModel(randomForest, X, Y, ...
        train, val);
    numPredictorsTrainError(i) = train_error;
    numPredictorsTrainError(i) = val_error;
end

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
