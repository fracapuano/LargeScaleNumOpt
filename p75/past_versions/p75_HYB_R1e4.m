clear
close all force
clc
%shutting down the warning related to the tolerance for the gmres resolution in the Inexact NM
warning("off") 

f = @(x) p75_function(x); 
gradf = @(x) p75_gradient(x); 
Hessf = @(x) p75_hessian(x); 

m_improv = 10; 
eta_thresh = 1e3; 
thresh_improv = 5e-2; 

kmax = 1e3; 
tolgrad = 1e-6;

%R10000
n = 4;
dim = 10^n; 

% assembling the starting point
x0 = [-1.2*ones(dim-1,1); -1]; %from here there is no convergence
% x0 = ones(dim, 1); 
% i = 1; 
% while i <= dim
%     x0(i) = -1.2; 
%     i = i + 2; 
% end

% Inexact Newton Method
load forcing_terms.mat

%backtracking parameters
btmax = 100; 
c1 = 1e-4; 
rho = 1e-2;

%inexact resolution parameters
gmres_maxit = 100; 
                                                                                    
[xk, fk, k, grads_out, values_out] = hybrid_PKplus_Newton_corr(...
    x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax, fterms_quad, gmres_maxit, m_improv, ...
    eta_thresh, thresh_improv); 
