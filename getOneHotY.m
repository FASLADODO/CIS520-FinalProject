function [ output ] = getOneHotY( Y )
%GETONEHOTY Assumes Y is 0/1
%   Detailed explanation goes here

output = full(ind2vec(Y' + 1));
end

