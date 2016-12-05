% Script to find optimal Isomap / SVM parameters

%% Initialization

% load data
load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

% setup variables
%distances = {'euclidean', 'seuclidean', 'cityblock', 'cosine'};
distances = {'euclidean'};
num_neighbors = 5:5:100;
dims = 100:100:2000;

errors = zeros(length(distances), length(num_neighbors), length(dims), 2);

% set up options for Isomap
options.display = 0;
options.overlay = 0;
options.verbose = 0;

%% Run trials

for i = 1:length(distances)
    % get distance matrix
    %D = squareform(pdist(X, distances{i}));
    D = L2_distance(X', X', 1);
    disp('Done calculating distance matrix!');
    for j = 1:length(num_neighbors)
        for d = 1:length(dims)
            % call isomap
            options.dims = 1:dims(d);
            X = Isomap(D, 'k', num_neighbors(j), options);
            % get SVM crossval error
            errors(i, j, d, :) = crossValError(@(X_train, Y_train, X_test) ...
                getYHatSVM(X_train, Y_train, X_test, 'linear', 1), X, Y, 3);
            
            % print progress
            printf('Distance function %s with %d neighbors and %d dimensions had %f training and %f val error\n',...
                distances{i}, num_neighbors(j), dims(d), ...
                errors(i, j, d, 1), errors(i, j, d, 2));
        end
    end
end


%% Plot results