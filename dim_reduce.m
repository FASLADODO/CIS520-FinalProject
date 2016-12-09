function [ X_projected, coeffs ] = dim_reduce( X, k, rebuild )
%dim_reduce Reduce dimensionality of X

k_max = min(size(X, 1), size(X, 2)) - 1;
if nargin < 2
    k = k_max;
end
if nargin < 3
    rebuild = false;
end

pca_filename = 'pca_coeffs.mat';
if rebuild | ~exist(pca_filename, 'file')
    coeffs = pca(X);
    save(pca_filename, 'coeffs');
else
    load(pca_filename);
end
X_projected = X * coeffs(:, 1:k);

% [W, H] = nnmf(X, k);
% X_projected = W*H;
% coeffs = 0;
end

