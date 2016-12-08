%% Load data
load('train_set/words_train.mat');

n = size(X, 1);
X_projected = dim_reduce(full(X), 500);

lambdas = linspace(.000001, .00001, 20);
lambdas = repmat(lambdas(9), 1, 5);
train_errors = zeros(size(lambdas));
val_errors = zeros(size(lambdas));

for i = 1:5
    fprintf('Lambda = %f\n', lambdas(i));
    %% Linear Shits
    [train_errors(i), val_errors(i)] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatLinear(X_train, Y_train, X_test, lambdas(i)), ...
        X_projected, Y, 10);

    fprintf('Train error: %f\n', train_errors(i));
    fprintf('Validation error: %f\n', val_errors(i));
end

i = length(lambdas);
plot(lambdas(1:i), train_errors(1:i), 'rx', ...
     lambdas(1:i), val_errors(1:i), 'b+');
% disp(mean(train_errors));
% disp(mean(val_errors));