function [xk, fk, k, grads, values] = SD_backtrack(x0, f, gradf, alpha0, ...
    kmax, tolgrad, c1, rho, btmax)

% Function that performs the steepest descent optimization method using
% backtracking as the technique for the selection of the steplenght alpha.
%
% Paramethers: 
% -----
% x0: n-dimensional column vector
% f: function handle related to the objective function
% gradf: function handle that describes the gradient of the objective function f
% alpha0: the initial steplength
% kmax: maximum number of iterations allowed
% tolgrad: value used as zero for the norm of the gradient
% c1: factor of the Armijo condition
% rho: ﻿fixed factor used to iteratevely down-scale alpha
% btmax: maximal number of backtracks allowed
%
% Returns:
% -----
% xk: the last x computed by the function;
% fk: the value of the objective function at the last point f(xk);
% gradfk_norm: value of the norm of the gradient at last point
% k: Number of iterations effectively run (k<=kmax)
% xseq: matrix storing the sequence {xk}_{k \in N}
% gradients: array storing the sequence of ||gradf(xk)|| for all k. 

% Function handle for the armijo condition 
farmijo = @(fk, alpha, gradfk, pk) ...
    fk + c1 * alpha * gradfk' * pk;

% Initializations
grads = zeros(1, kmax); 
values = zeros(1, kmax);

xk = x0;
fk = f(xk);
gradfk = gradf(xk);
k = 1;
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
    grads(k) = gradfk_norm;
    values(k) = fk;
    
    % Increase the step by one
    k = k + 1;
    
end
k = k - 1; 

%properly return the sequence of gradients
grads = grads(1:k); values = values(1:k);
end