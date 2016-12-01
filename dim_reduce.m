function [ X_projected, coeffs ] = dim_reduce( X, k )
%dim_reduce Reduce dimensionality of X

if nargin == 1
    k = size(X, 1) - 1;
end

coeffs = pca(X, 'NumComponents', k);
X_projected = X * coeffs;

end

