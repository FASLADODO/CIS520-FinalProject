%% Logistic Regression Tuning
lambdas = linspace(1.5e-4, 5e-4, 10);
bestValError = Inf;
bestLambda = 0;
trainErrors = zeros(1, length(lambdas));
valErrors = zeros(1, length(lambdas));
for i = 1:length(lambdas)
    fprintf('Lambda = %f\n', lambdas(i));
    [train_error, val_error] = crossValError(@(X_train, Y_train, X_test) ...
        getYHatLinear(X_train, Y_train, X_test, lambdas(i), 'logistic'), ...
        X_projected, Y, 10);

    fprintf('Train error: %f\n', train_error);
    fprintf('Validation error: %f\n', val_error);
    trainErrors(i) = train_error;
    valErrors(i) = val_error;
    
    if val_error < bestValError
        fprintf('New high score of %d\n', val_error)
        bestValError = val_error;
        bestLambda = lambdas(i);
    end
end
plot(lambdas, trainErrors, 'rx', ...
    lambdas, valErrors, 'b+');
