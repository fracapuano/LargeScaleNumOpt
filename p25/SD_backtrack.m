function [xk, fk, k, grads, values] = SD_backtrack(x0, f, gradf, alpha0, ...
    kmax, tolgrad, c1, rho, btmax)

% Function that performs the steepest descent optimization method using
% backtracking as the technique for the selection of the steplenght alpha.
%
% INPUTS: 
% x0: n-dimensional column vector
% f: function handle that describes a function R^n->R
% gradf: function handle that computes the gradient of f
% alpha0: the initial steplength
% kmax: maximum number of iterations allowed
% tolgrad: value used to declare null the norm of the gradient
% c1, rho: ﻿scalars used to evaluate the Armijo condition
% btmax: maximal number of backtrack its to scale down alpha_k
%
% OUTPUTS:
% xk: the last x computed by the function;
% fk: the value f(xk);
% gradfk_norm: value of the norm gradf(xk)
% k: Index of the last iteration performed
% gradients: sequence of the norm of the gradients 

% Function handle for the armijo condition
farmijo = @(fk, alpha, gradfk, pk) ...
    fk + c1 * alpha * gradfk' * pk;

% Initializations
grads = zeros(1, kmax); 
values = zeros(1, kmax);

xk = x0;
fk = f(xk);
gradfk = gradf(xk);
k = 0;
gradfk_norm = norm(gradfk);

while k < kmax && gradfk_norm >= tolgrad
    % Compute the descent direction
    pk = -gradf(xk);
    
    % Reset the value of alpha
    alpha = alpha0;
    
    % Compute the candidate new xk
    xnew = xk + alpha * pk;
    % Compute the value of f in the candidate new xk
    fnew = f(xnew);
    
    bt = 0;
    % Backtracking strategy: 
    while bt < btmax && fnew > farmijo(fk, alpha, gradfk, pk)
        % Reduce the value of alpha
        alpha = rho * alpha;
        % Update xnew and fnew
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
    end
    
    % Update xk, fk, gradfk_norm
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    % Increase the step by one
    k = k + 1;
    
    % Store gradient and objective function values in respective arrays
    grads(k) = gradfk_norm; values(k) = fk;
    
end

% Cut the array properly to return results
grads = grads(1:k); values = values(1:k);
end