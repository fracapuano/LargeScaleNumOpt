function Hessf = p59_hessian(f, x, h)
%P59_HESSIAN computes numerically the Hessian of the function defined in
%Problem 59 exploiting its sparsity pattern

n = length(x);
main_diag = zeros(n,1); 
first_sub_diag = zeros(n,1); 
second_sub_diag = 2*ones(n,1); 
I = speye(n); 

fx = f(x); 
parfor i=1:n
    main_diag(i) = (f(x + h * I(:,i)) - 2*fx + f(x - h * I(:,i)))/h^2;
    if i < n
        first_sub_diag(i) = (f(x+h*I(:,i)+h*I(:,i+1)) - f(x+h*I(:,i)) - f(x+h*I(:,i+1)) + fx)/h^2; 
    end
end

lower = spdiags([second_sub_diag first_sub_diag main_diag], -2:0, n, n);
upper = (spdiags([second_sub_diag first_sub_diag], [-2 -1], n, n))';

Hessf = lower + upper; 
