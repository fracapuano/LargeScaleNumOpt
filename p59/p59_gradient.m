function grad = p59_gradient(x, h, type)
%P59_GRADIENT returns the numerical gradient of the function defined in
%problem 59 exploiting its sparsity pattern.

n = length(x); 
grad = zeros(n,1); 
subgradient = @(x,i) p59_subgradient(x, h, i, type);

for i=1:n
    grad = grad + p59_subfunction(x,i) * subgradient(x,i);
end

end

