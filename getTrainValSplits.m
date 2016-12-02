function [ train, val ] = getTrainValSplits( n, trainProportion )
%getTrainValSplits
%   n - number of training examples
%   trainProportion (optional) - proportion of data to be used for training

if nargin < 2
    trainProportion = 0.9;
end

randPerm = randperm(n);
train = randPerm(1:floor(trainProportion * n));
val = randPerm(floor(trainProportion * n) + 1:end);

end

