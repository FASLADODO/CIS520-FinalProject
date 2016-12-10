function [ yhat_train, yhat_test ] = getYHatNN( X_train, Y_train, X_test )
%GETYHATNN Summary of this function goes here
%   Detailed explanation goes here

Ynew = getOneHotY(Y_train);
net = patternnet(500);
net.trainFcn = 'trainscg';
net.performParam.regularization = 0.1;
net = train(net, X_train', Ynew);

yhat_train = (vec2ind(net(X_train')) - 1)';

yhat_test = (vec2ind(net(X_test')) - 1)';

end

