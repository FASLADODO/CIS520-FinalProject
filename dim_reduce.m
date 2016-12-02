function [ X_projected, coeffs ] = dim_reduce( X, k, rebuild )
%dim_reduce Reduce dimensionality of X

k_max = size(X, 2);
if nargin < 2
    k = k_max;
end
if nargin < 3
    rebuild = false;
end

pca_filename = 'pca_coeffs.mat';
if rebuild || ~exist(pca_filename, 'file')
    coeffs = pca(X);
    save(pca_filename, 'coeffs');
else
    coeffs = load(pca_filename);
end
X_projected = X * coeffs(:, 1:k);

end

