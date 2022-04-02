function [xk, fk, gradfk_norm, k, xseq, grads, values, inner_its] = innewton_general_with_correction(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax, fterms, pcg_maxit)

% Function that performs the newton optimization method, 
% implementing the backtracking strategy and, optionally, finite
% differences approximations for the gradient and/or the Hessian.
%
% INPUTS:
% x0 = n-dimensional column vector;
% f = function handle that describes a function R^n->R;
% gradf = function handle that describes the gradient of f (not necessarily
% used);
% Hessf = function handle that describes the Hessian of f (not necessarily 
% used);
% kmax = maximum number of iterations permitted;
% tolgrad = value used as stopping criterion w.r.t. the norm of the
% gradient;
% c1 = ﻿the factor of the Armijo condition that must be a scalar in (0,1);
% rho = ﻿fixed factor, lesser than 1, used for reducing alpha0;
% btmax = ﻿maximum number of steps for updating alpha during the 
% backtracking strategy;
% fterms = f. handle "@(gradfk, k) ..." that returns the forcing term
% eta_k at each iteration
% pcg_maxit = maximum number of iterations for the pcg solver.
%
% OUTPUTS:
% xk = the last x computed by the function;
% fk = the value f(xk);
% gradfk_norm = value of the norm of gradf(xk)
% k = index of the last iteration performed
% xseq = n-by-k matrix where the columns are the xk computed during the 
% iterations
% btseq = 1-by-k vector where elements are the number of backtracking
% iterations at each optimization step.
%

% Function handle for the armijo condition
farmijo = @(fk, alpha, gradfk, pk) ...
    fk + c1 * alpha * gradfk' * pk;

% Initializations
xseq = zeros(length(x0), kmax);
grads = zeros(1, kmax);
values = zeros(1, kmax); 
inner_its = zeros(1, kmax); 

xk = x0;
n = length(x0);
fk = f(xk);
k = 0;
gradfk = gradf(xk);
gradfk_norm = norm(gradfk);

%tau initialization
beta = 1e-3; 
a_min = min(diag(Hessf(xk))); 

if a_min > 0
    tau_zero = 0; 
else 
    tau_zero = beta - a_min; 
end %tau initialization

spd_max = 50;
tau_k = tau_zero;

while k < kmax && gradfk_norm >= tolgrad
    % "INEXACTLY" compute the descent direction as solution of
    Hessf_computed = Hessf(xk);
    
    % TOLERANCE VARYING W.R.T. FORCING TERMS:
    epsilon_k = fterms(gradfk, k) * norm(gradfk);

    % CHECK IF THE HESSIAN IS SPD     
    spd_it = 0;
    
    try chol(Hessf_computed);
        
        Hessf_computed = Hessf_computed; %when the hessian is spd don't touch it

    catch ME %the hessian is not SPD

        while spd_it <= spd_max
            Hessf_SPD = Hessf_computed + tau_k * speye(n);
            try chol(Hessf_SPD);
                %disp("HESSIAN IS SPD")%Hessian is SPD
                tau_k = tau_zero; 
                Hessf_computed = Hessf_SPD;
                break
    
            catch ME %Hessian is not SPD. Incrementing the value of tau_k
                %disp("HESSIAN IS NOT SPD. Adding identity.")
                tau_k = max(beta, 2*tau_k);
                spd_it = spd_it + 1;
                continue
    
            end
        end
    end

    % INEXACTLY COMPUTED DESCENT DIRECTION
    [pk, ~, ~, its] = gmres(Hessf_computed, -gradfk, [], epsilon_k, pcg_maxit);
  
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
    
    % Store current xk in xseq
    xseq(:, k) = xk;

    % Store gradient and function values in respective arrays
    grads(k) = gradfk_norm; values(k) = fk; inner_its(k) = its; 
    
end

% "Cut" xseq and btseq to the correct size
xseq = xseq(:, 1:k);
grads = grads(1:k); values = values(1:k);

end