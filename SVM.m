load('train_set/words_train.mat');
load('train_set/train_img_prob.mat');
load('train_set/train_cnn_feat.mat');
load('train_set/train_color');

N = size(X, 1);

for numComponents = 100:100:1500
    disp(numComponents);
    Xnew = full(X);
    Xnew = dim_reduce(Xnew, numComponents);

    numFolds = 10;
    indices = crossvalind('Kfold', N, 10);
    cp = classperf(Y);
    train_errors = ones(numFolds, 1);
    test_errors = ones(numFolds, 1);
    for i = 1:numFolds
        test = (indices == i); 
        train = ~test;
        Mdl = fitcsvm(Xnew(train, :), Y(train));
        [train_errors(i), test_errors(i)] = evaluateModel(Mdl, Xnew, Y, train, test);
    end

    disp(mean(train_errors));
    disp(mean(test_errors));
%     plot(1:10, train_errors, 'rx', 1:10, test_errors, 'b+');
end
