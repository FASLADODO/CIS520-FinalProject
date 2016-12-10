function [ C ] = generateConfusionMatrix( y, yhat )
%GENERATECONFUSIONMATRIX 
%   C(i,j) is the accuracy of observations actually in group i but predicted 
%   to be in group j

C = zeros(2, 2);

zeros_idx = y == 0;
ones_idx = y == 1;

C(1, 1) = sum(y(zeros_idx) == yhat(zeros_idx));
C(1, 2) = sum(y(zeros_idx) ~= yhat(zeros_idx));
C(2, 1) = sum(y(ones_idx) ~= yhat(ones_idx));
C(2, 2) = sum(y(ones_idx) == yhat(ones_idx));
end

