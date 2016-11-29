function [ X_projected, coeffs ] = dim_reduce( X, k )
%dim_reduce Reduce dimensionality of X

if nargin == 1
    k = 300;
end

% [~, X_projected] = pca(X, 'NumComponents', k);
coeffs = pca(X, 'NumComponents', k);
X_projected = X*coeffs;
end

