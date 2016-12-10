function [ randomForest ] = RandomForestBayesianTuning( X, Y )
%random_forest Trains a random forest
%   Trains a random forest with Bayesian optimized hyperparameters:
%   - leaf size
%   - predictors sampled at each node

rng('default'); % For reproducibility
[n , p] = size(X);

% Get train/test splits
trainProportion = 0.7;
randPerm = randperm(n);
train = randPerm(1:floor(trainProportion * n));
test = randPerm(floor(trainProportion * n) + 1:end);

% Specify hyperparameters to be tuned
maxMinLeafSize = 20;
minLeafSize = optimizableVariable('minLeafSize', [1, maxMinLeafSize], ... 
    'Type', 'integer');
numPredictors = optimizableVariable('numPredictors', [1, p - 1], ...
    'Type', 'integer');
hyperparametersRF = [minLeafSize; numPredictors];

% Optimize hyperparameters
results = bayesopt(@(params) oobErrRF(params, X(train, :), Y(train, :)), hyperparametersRF, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', 'Verbose', 0);
bestHyperparameters = results.XAtMinObjective;

% Train best random forest
randomForest = TreeBagger(128, X(train, :), Y(train, :), ...
    'MinLeafSize', bestHyperparameters.minLeafSize,...
    'NumPredictorstoSample', bestHyperparameters.numPredictors);

% Evaluate best random forest
[train_error, test_error] = evaluateModel(randomForest, X, Y, train, test);
fprintf('Training Error: %d', train_error)
fprintf('Test Error: %d', test_error)

end

function oobErr = oobErrRF(params, X, Y)
%oobErrRF Trains random forest and estimates out-of-bag quantile error
%   oobErr trains a random forest of 128 trees using the predictor data in 
%   X and the parameter specification in params, and then returns the 
%   out-of-bag quantile error based on the median.
%   Params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
randomForest = TreeBagger(128, X, Y, ...
    'OOBPrediction', 'on', 'MinLeafSize', params.minLeafSize,...
    'NumPredictorstoSample', params.numPredictors);
oobErr = oobQuantileError(randomForest);
end
