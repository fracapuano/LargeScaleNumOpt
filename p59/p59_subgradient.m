function grad = p59_subgradient(x, h, j, type)
%P59_GRADIENT computes numerically the gradient of the sub_functions defined in
%Problem 59 exploiting the tridiagonal sparsity pattern

n = length(x); 
grad = zeros(n,1);
I = speye(n); 
f = @(x) p59_subfunction(x, j); 
fx = f(x); 

switch type
    case "fw"
        switch j
            case 1
                grad(1) = (f(x + h*I(:,1)) - fx)/h; 
                grad(2) = (f(x + h*I(:,2)) - fx)/h; 
            case n
                grad(n-1) = (f(x+h*I(:,n-1)) - fx)/h; 
                grad(n) = (f(x+h*I(:,n)) - fx)/h; 
            otherwise
                grad(j-1) = (f(x+h*I(:,j-1)) - fx)/h;
                grad(j) = (f(x+h*I(:,j)) - fx)/h;
                grad(j+1) = (f(x+h*I(:,j+1)) - fx)/h;
                
        end
    case "c"
        switch j
            case 1
                grad(1) = (f(x+h*I(:,1))-f(x-h*I(:,1)))/(2*h); 
                grad(2) = (f(x+h*I(:,2))-f(x-h*I(:,2)))/(2*h); 
            case n
                grad(n-1) = (f(x+h*I(:,n-1))-f(x-h*I(:,n-1)))/(2*h); 
                grad(n) = (f(x+h*I(:,n))-f(x-h*I(:,n)))/(2*h); 
            otherwise
                grad(j-1) = (f(x+h*I(:,j-1))-f(x-h*I(:,j-1)))/(2*h);
                grad(j) = (f(x+h*I(:,j))-f(x-h*I(:,j)))/(2*h);
                grad(j+1) = (f(x+h*I(:,j+1))-f(x-h*I(:,j+1)))/(2*h);  
        end
end %main switch
grad = sparse(grad); 
end
