% Script to find optimal RBF parameters

%% Initialization

%% Add paths
addpath(genpath('../'));
addpath(genpath('../Utils/'));

% load data
load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

% call PCA
Xt = dim_reduce(full(X), 500);

% setup variables
widths = 10.^(-3:1:3);
boxes = [10.^(-3:1:3) Inf];
priors = {'empirical', 'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 1;

%% Run initial trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 3);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end

%% Run finer trials

% setup variables
widths = [1 3 7 10 25 50 75 100];
boxes = [1 5 10 30 60 100 300 600 1000 1300 1500];
priors = {'empirical', 'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 1;

% run trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 3);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end


%% Find 2-4 opt trials

% setup variables
widths = 2:0.25:4;
boxes = [0.5:0.1:1 1:0.5:6];
priors = {'empirical', 'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 1;

% run trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 3);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end

%% Find optimum for 3.75 scale

% setup variables
widths = 3.75;
boxes = 2:0.1:3;
priors = {'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 5;

% run trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                fprintf('Trial %d\n', t);
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 6);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end

% this achieved a minimum 0.205822 6-fold validation error over 5 trials,
% for a box size of 2.8

%% Find optimum for 4.00 scale

% setup variables
widths = 4.00;
boxes = 1:0.2:3.5;
priors = {'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 5;

% run trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                fprintf('Trial %d\n', t);
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 6);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end

% this achieved three low points for 6-fold validation error over 5 trials,
% at 0.205733 for 1.8 box size, 0.205467 for 2.4 box size, and 0.205556 for
% 3.2 box size.  Further testing around these regions:

%% Testing around optimum for 4.00 scale

% setup variables
widths = 4.00;
boxes = [1.7 1.9 2.3 2.5 3.1 3.3];
priors = {'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 5;

% run trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                fprintf('Trial %d\n', t);
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 6);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end

% none of these achieved lower than 0.205956 error, the above is the
% minimum

%% RBF Submission 1

[Xnew, coeffs] = dim_reduce(X, 500);
Mdl = fitcsvm(Xnew, Y, 'KernelFunction', 'rbf', 'BoxConstraint', 2.4, ...
    'KernelScale', 4.00, 'Prior', 'uniform');
save('SVM_Model.mat', 'Mdl', 'coeffs');

% Note: increasing dimensions from PCA to 900 improved validation error to
% 0.200267, will try this as next submission

%% Find 10-40 opt trials

% setup variables
widths = 10:5:40;
boxes = [7 10 16 22 30 40 50 60 73 87 100 150 200 250 300 350 400];
priors = {'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 1;

% run trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 3);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end

%% Find optimum for 20.00 scale

% setup variables
widths = 20.00;
boxes = [46.5:0.25:47.5 51.5:0.25:52.5 81.5:0.25:82.5];
% 47, 52, and 82 achieved lowest 3-fold single run in the range 40-87
priors = {'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 5;

% run trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                fprintf('Trial %d\n', t);
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 6);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end

%% Finer optimum for 20.00 scale

% setup variables
widths = 20.00;
boxes = [46.7 46.8 47.75 51.9 52.1 52.4 52.6];
priors = {'uniform'};

train_errors = zeros(length(widths), length(boxes), length(priors));
val_errors = zeros(length(widths), length(boxes), length(priors));
num_trials = 5;

% run trials
for p = 1:length(priors)
    fprintf('********************************************\n');
    fprintf('Now using a %s prior\n', priors{p});
    fprintf('********************************************\n');
    for w = 1:length(widths)
        for b = 1:length(boxes)
            for t = 1:num_trials
                fprintf('Trial %d\n', t);
                % get SVM crossval error
                [err1, err2] = crossValError(@(X_train, Y_train, X_test) ...
                    getYHatSVM(X_train, Y_train, X_test, 'rbf', boxes(b), widths(w), priors{p}),...
                    Xt, Y, 6);
                train_errors(w, b, p) = train_errors(w, b, p) + err1;
                val_errors(w, b, p) = val_errors(w, b, p) + err2;
            end

            %means
            train_errors(w, b, p) = train_errors(w, b, p) / num_trials;
            val_errors(w, b, p) = val_errors(w, b, p) / num_trials;

            % print progress
            fprintf('RBF with %f kernel scale and box size %f had %f training and %f val error\n',...
                widths(w), boxes(b), train_errors(w, b, p), val_errors(w, b, p));
        end
    end
end

% this achieved a low point for 6-fold validation error over 5 trials
% at 0.198622 for 46.75 box size

% NOTE: increasing dimensions from PCA to 800 improved validation error to
% 0.192111 (on 2 trials), which will be third submission