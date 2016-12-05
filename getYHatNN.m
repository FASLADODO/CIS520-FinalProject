function [ yhat_train, yhat_test ] = getYHatNN( X_train, Y_train, X_test )
%GETYHATNN Summary of this function goes here
%   Detailed explanation goes here

Ynew = getOneHotY(Y_train);
net = pattern_net(500);
net = train(net, X_train', Ynew');

yhat_train = (vec2ind(net(X_train')) - 1)';

yhat_test = (vec2ind(net(X_test')) - 1)';

end

