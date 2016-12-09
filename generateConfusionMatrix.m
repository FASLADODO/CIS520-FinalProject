function [ C ] = generateConfusionMatrix( y, yhat )
%GENERATECONFUSIONMATRIX 
%   C(i,j) is the accuracy of observations actually in group i but predicted 
%   to be in group j

C = zeros(2, 2);

zeros_idx = y == 0;
ones_idx = y == 1;
num_zeros = sum(zeros_idx);
num_ones = sum(ones_idx);

C(1, 1) = sum(y(zeros_idx) == yhat(zeros_idx)) / num_zeros;
C(1, 2) = sum(y(zeros_idx) ~= yhat(zeros_idx)) / num_zeros;
C(2, 1) = sum(y(ones_idx) ~= yhat(ones_idx)) / num_ones;
C(2, 2) = sum(y(ones_idx) == yhat(ones_idx)) / num_ones;
end

