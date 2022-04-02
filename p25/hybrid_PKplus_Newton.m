function [xk, fk, k, k_start, grads_out, values_out, inner_its] = ...
         hybrid_PKplus_Newton(...
         x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax, ...
         fterms, gmres_maxit, restart, perc)

% Function that combines Polak-Ribière CGM and Inexact Newton method. It is
% basically running PK+ until a certain iterate khat = perc * kmax to then switch
% to Newton Method.

%
% INPUTS: 
% x0: n-dimensional column vector
% f: function handle that describes a function R^n->R
% gradf: function handle that computes the gradient of f
% Hessf: function handle that computes the Hessian of f
% alpha0: the initial steplength
% kmax: maximum number of iterations allowed
% tolgrad: value used to declare null the norm of the gradient
% c1, rho: ﻿scalars used to evaluate the Armijo condition
% btmax: maximal number of backtrack its to scale down alpha_k
% gmres_maxit: maximum number of iterations for the gmres solver
% restart: restart parameter for the gmres solver
% perc: the percentage on kmax that triggers the switch to Inexact Newton Method
% 
% OUTPUTS:
% xk: the last x computed by the function;
% fk: the value f(xk);
% gradfk_norm: value of the norm gradf(xk)
% k: index of the last iteration performed
% grads_out: sequence of the norm of the gradients
% values_out: sequence of the objective function values
% inner_its: sequence of the inner iterations used by the gmres solver to
%            converge to solutions

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

grads = zeros(1, kmax); 
values = zeros(1, kmax); 

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
    while bt < btmax && fnew > farmijo(fk, alpha_k, gradfk, pk)
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
    
    if k > floor(perc * kmax) % switching to Inexact Newton Method
        break
    end
end

k_start = k - 1; %Newton starts at the last k of PK+

[xk, fk, ~, k_N, grads_N, values_N, inner_its] = ... 
                innewton_general(xk, f, gradf, Hessf, (kmax - k_start), ...
                                tolgrad, c1, rho, btmax, fterms, gmres_maxit, restart);

k = k_start + k_N; 

grads_1 = grads(1:k_start); 
values_1 = values(1:k_start); 

grads_out = [grads_1; grads_N']; 
values_out = [values_1; values_N']; 

k = k_start + k_N; 
end


    
