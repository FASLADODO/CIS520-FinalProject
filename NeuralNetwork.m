clear
%% Setup Cortexsys
addpath('Cortexsys/nn_gui');
addpath('Cortexsys/nn_core');
addpath('Cortexsys/nn_core/cuda');
addpath('Cortexsys/nn_core/mmx');
addpath('Cortexsys/nn_core/Optimizers');
addpath('Cortexsys/nn_core/Activations');
addpath('Cortexsys/nn_core/Wrappers');
addpath('Cortexsys/nn_core/ConvNet');

PRECISION = 'double';
useGPU = false;
whichGPU = [];
plotOn = true;
defs = definitions(PRECISION, useGPU, whichGPU, plotOn);

%% Load and setup data
TRAIN_PROPORTION = 0.9;
NUM_CLASSES = 2;

load('train_set/words_train.mat');
X = X'; % put examples in columns because that's what the network expects
[p, n] = size(X);

% Change Y from 0-1 to 1-2 vector
Y(Y == 1) = 2;
Y(Y == 0) = 1;
Y = full(Y);

% Split into training and validation sets
% (lowercase 'y' denotes that it is a vector)
rng('default')
[train, val] = getTrainValSplits(n, TRAIN_PROPORTION);
X_train = X(train, :);
y_train = Y(train, :);
X_val = X(val, :);
y_val = Y(val, :);

% Get one-hot represenation of Y
% (uppercase 'Y' denotes that it is a matrix)
oneHotVectors = speye(NUM_CLASSES);
Y_train = oneHotVectors(y_train, :)';
Y_val = oneHotVectors(y_val, :)';

% Finish getting data into right format for network
X_train = varObj(precision(X_train, defs), defs, defs.TYPES.INPUT);
Y_train = varObj(Y_train, defs, defs.TYPES.OUTPUT);
X_val = varObj(precision(X_val, defs), defs, defs.TYPES.INPUT);
Y_val = varObj(Y_val, defs, defs.TYPES.OUTPUT);

%% Setup network architecture
HIDDEN_LAYER_SIZE = 256;
layers.af{1} = [];
layers.sz{1} = [p 1 1];
layers.typ{1} = defs.TYPES.INPUT;

layers.af{end+1} = LReLU(defs, defs.COSTS.SQUARED_ERROR);
layers.sz{end+1} = [HIDDEN_LAYER_SIZE 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

layers.af{end+1} = softmax(defs, defs.COSTS.CROSS_ENTROPY);
layers.sz{end+1} = [NUM_CLASSES 1 1];
layers.typ{end+1} = defs.TYPES.FULLY_CONNECTED;

if defs.plotOn
    nnShow(1, layers, defs);
end

%% Setup network parameters
% Training parameters
params.maxIter = 1000; % How many iterations to train for
params.miniBatchSize = 128; % set size of mini-batches**

% Regularization parameters
params.maxnorm = 0; % enable max norm regularization if ~= 0
params.lambda = 1; % enable L2 regularization if ~= 0
% params.alphaTau = 0.25*params.maxIter; % Learning rate decay
% params.denoise = 0.25; % enable input denoising if ~= 0
% params.dropout = 0.6; % enable dropout regularization if ~= 0
params.tieWeights = false; % enable tied weights for autoencoder?
params.beta_s = 0; % Strength of sparsity penalty; set to 0 to disable
params.rho_s0 = 0; % Target hidden unit activation for sparsity penalty

% Learning rate parameters
params.momentum = 0.9; % Momentum for stochastic gradient descent
params.alpha = 0.01; % Learning rate for SGD
params.rho = 0.95; % AdaDelta hyperparameter
params.eps = 1e-6; % AdaDelta hyperparameter

% Conjugate gradient parameters
params.cg.N = 10; % Max CG iterations before reset
params.cg.sigma0 = 0.01; % CG Secant line search parameter
params.cg.jmax = 10; % Maximum CG Secant iterations
params.cg.eps = 1e-4; % Update threshold for CG
params.cg.mbIters = 10; % How many CG iterations per minibatch?

%% Initialize network
nn = nnLayers(params, layers, X_train, y_train, {}, {}, defs);
nn.initWeightsBiases();

%% Train network
costFunc = @(nn, r, newRandGen) nnCostFunctionCNN(nn, r, newRandGen);
nn = gradientDescentAdaDelta(costFunc, nn, defs, X_val, Y_val, y_val, ...
    y_train, 'Training Entire Network');

% y: train y ("trainError(end+1) = 100 - mean(double(pred == y(r))) * 100;")
% yts: test y ("testError(end+1) = 100 - mean(double(pred == yts)) * 100;")
% Yts: also test y? (used for computing test set error)
% Xts: test y (used for computing test set error)
