function plotTrainValError( x, trainError, valError, ...
    titleStr, xStr )
%plotTrainValError Plot the training and validation error
%   optional arguments:
%       titleStr - defaults to ''
%       xStr - defaults to variable name of x

if nargin < 4
    titleStr = '';
end
if nargin < 5 
    xStr = inputname(1);
end

figure
hold on
plot(x, trainError, 'rx')
plot(x, valError, 'bx')
legend('Train Error', 'Val Error')
title(titleStr)
xlabel(xStr)
ylabel('Error')
hold off

end

