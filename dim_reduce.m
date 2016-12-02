function [ X_projected, coeffs ] = dim_reduce( X, k, rebuild )
%dim_reduce Reduce dimensionality of X

if nargin < 2
    k = size(X, 1) - 1;
end
if nargin < 3
    rebuild = false;
end

pca_filename = 'pca_coeffs.mat';
if rebuild || ~exist(pca_filename, 'file')
    coeffs = pca(X, 'NumComponents', k);
    save(pca_filename, 'coeffs')
else
    coeffs = load(pca_filename);
end
X_projected = X * coeffs;

end

