function [xk, fk, k, grads, values] = ...
         nonlinear_PKplus(f, gradf, tolgrad, x0, kmax, c1, rho, max_bt)

% Function implementing the Polak-Ribiere "+" Variant of
% the Non Linear Conjugated Gradient Method proposed originally by Reeves.
%
% INPUTS: 
% f: function handle that describes a function R^n->R
% gradf: function handle that computes the gradient of f
% tolgrad: value used to declare null the norm of the gradient
% x0: n-dimensional column vector.
% kmax: maximal number of iterations allowed
% c1, rho: parameters for the backtracking strategy 
% max_bt: maximum number of backtrack its to scale down alpha_k

% OUTPUTS: 
% xk: the last x computed by the function 
% fk: the value f(xk)
% k: index of the last iteration performed 
% grads: sequence of the norm of the gradients
% values: sequence of the objective function values

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
    NewGradNorm = norm(SD_New);
    k = k + 1;
    
end
k = k - 1;

% Cut the array properly to return results
fk = f(xk); grads = grads(1:k); values = values(1:k); 
end
