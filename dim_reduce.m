function [ X_projected ] = dim_reduce( X, k )
%dim_reduce Reduce dimensionality of X

if nargin == 1
    k = 300;
end

[~, X_projected] = pca(X, 'NumComponents', k);

end

