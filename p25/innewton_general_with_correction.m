function [xk, fk, gradfk_norm, k, grads, values, inner_its] = ...
    innewton_general_with_correction(x0, f, gradf, Hessf, hess_corrections, kmax, ...
    tolgrad, c1, rho, btmax, fterms, gmres_maxit, restart)

% Function that performs the newton optimization method, 
% implementing the backtracking strategy and, optionally, finite
% differences approximations for the gradient and/or the Hessian.
%
% INPUTS:
% x0: n-dimensional column vector;
% f: function handle that describes a function R^n->R;
% gradf: function handle that computes the gradient of f 
% hess_corrections: maximal number of iterations used to "correct" the
%                   hessian in case is not SPD
% Hessf: function handle that describes the Hessian of f
% kmax: maximum number of iterations permitted;
% tolgrad: value used to declare null the norm of the gradient
%
% c1, rho: scalars used to evaluate the Armijo condition
% btmax: ﻿maximum number of backtrack its to scale down alpha_k.
% fterms: function handle that returns the forcing terms
% gmres_maxit: maximum number of iterations for the gmres solver.
% restart: restart parameter for the gmres solver
%
% OUTPUTS:
% xk: the last x computed by the function;
% fk: the value f(xk);
% gradfk_norm: value of the norm of gradf(xk)
% k: index of the last iteration performed
% grads: sequence of the norm of the gradients
% values: sequence of the objective function values
% inner_its: sequence of the inner iterations used by the gmres solver to
%            converge to solutions

% Function handle for the armijo condition
farmijo = @(fk, alpha, gradfk, pk) ...
    fk + c1 * alpha * gradfk' * pk;

% Initializations
grads = zeros(1, kmax);
values = zeros(1, kmax); 
inner_its = zeros(1, kmax); 

xk = x0;
n = length(x0);
fk = f(xk);
k = 0;
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);

% tau initialization
beta = 1e-3; 
a_min = min(diag(Hessf(xk))); 

if a_min > 0
    tau_zero = 0; 
else 
    tau_zero = beta - a_min; 
end %tau initialization

spd_max = hess_correction;
tau_k = tau_zero;

while k < kmax && gradfk_norm >= tolgrad

    % "INEXACTLY" compute the descent direction as solution of
    Hessf_computed = Hessf(xk);
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    % TOLERANCE VARYING W.R.T. FORCING TERMS:
    epsilon_k = fterms(gradfk_norm) * gradfk_norm;

    % CHECK IF THE HESSIAN IS SPD     
    spd_it = 0;
    
    try chol(Hessf_computed); % when the Hessian is spd, don't fix it
        Hessf_computed = Hessf_computed; 
    catch ME % i.e. the hessian is not SPD
        while spd_it <= spd_max
            Hessf_SPD = Hessf_computed + tau_k * speye(n); %updating the Hessian
            try chol(Hessf_SPD);
                
                tau_k = tau_zero; 
                Hessf_computed = Hessf_SPD;
                break
    
            catch ME %Hessian is not SPD. Incrementing the value of tau_k
                
                tau_k = max(beta, 2*tau_k);
                spd_it = spd_it + 1;
                continue
    
            end
        end
    end

    % INEXACTLY COMPUTED DESCENT DIRECTION
    [pk, ~, ~, its] = gmres(Hessf_computed, -gradfk, [], epsilon_k, gmres_maxit);
  
    % Reset the value of alpha
    alpha = 1;
    
    % Compute the candidate new xk
    xnew = xk + alpha * pk;

    % Compute the value of f in the candidate new xk
    fnew = f(xnew);
    
    bt = 0;
    % Backtracking strategy: 
    % 2nd condition is the Armijo condition not satisfied
    while bt < btmax && fnew > farmijo(fk, alpha, xk, pk)
        % Reduce the value of alpha
        alpha = rho * alpha;

        % Update xnew and fnew w.r.t. the reduced alpha
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

    % Store gradient, function values and inner its in respective arrays
    grads(k) = gradfk_norm; values(k) = fk; inner_its(k) = its(2); 
    
end

% Cut the array properly to return results
grads = grads(1:k); values = values(1:k); inner_its = inner_its(1:k); 

end