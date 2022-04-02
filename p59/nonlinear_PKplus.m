function [xk, fk, k, grads, values] = nonlinear_PKplus(f, gradf, tolgrad, x0, kmax, c1, rho, max_bt)
%NONLINEAR_PKPLUS is a function implementing the Polak-Ribiere Variant of
%the Non Linear Conjugated Gradient Method proposed originally by Reeves.
%
%   Paramethers: 
%   -----------
%   f: function handle representing the function to actually minimize
%   gradf: function handle representing the gradient of the objective
%          function
%   tolgrad: a scalar to conclude whether or not the gradient is small
%            enough
%   x0: a starting point for the method.
%   kmax: maximal number of iterations allowed
%   c1, rho: parameters for the backtracking strategy 
%   max_bt: maximal number of backtrack iterations

%   Returns: 
%   -----------
%   xk: the optimal point 
%   fk: the value of the function at the optimal
%   k: the number of iterations necessary to get to the optimal point.
%   grads: array storing the norm of the gradient at each iteration
%   values: array storing the value of the objective function at each
%           iteration

xk = x0; 
gradf_xk = gradf(xk); 
alpha_0 = 1; 

% Function handle for the armijo condition
farmijo = @(fk, alpha, gradfk, pk) ...
    fk + c1 * alpha * gradfk' * pk;

alpha_k = alpha_0;
SD_Old = -gradf_xk;
p0 = SD_Old;
k = 1; 

grads = zeros(kmax, 1); 
values = zeros(kmax, 1); 
NewGradNorm = norm(gradf_xk);

pk = p0;
while k<kmax && NewGradNorm > tolgrad
    
    gradfk = gradf(xk);
    
    fk = f(xk);
    grads(k) = norm(gradfk);
    values(k) = fk;

    SD_New = -gradfk; 
    betaPR_k = (SD_New'*(SD_New-SD_Old))/(SD_Old'*SD_Old);

    %implementing the '+' variation
    betaPR_k = max(betaPR_k, 0); 

    pk = SD_New + betaPR_k*pk;
    alpha_k = alpha_0;
    %candidate new point
    xnew = xk + alpha_k * pk;
    fnew = f(xnew);
    
    bt = 0;
    % Backtracking strategy
    while bt < max_bt && fnew > farmijo(fk, alpha_k, gradfk, pk)
        % Reduce the value of alpha
        alpha_k = rho * alpha_k;
        % recompute the new point
        xnew = xk + alpha_k * pk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
    end
    
    xk = xk + alpha_k*pk;

    SD_Old = SD_New;
    NewGradNorm = norm(SD_New)
    k = k + 1;
    
end
k = k - 1;

fk = f(xk); 
grads = grads(1:k);
values = values(1:k); 
end
