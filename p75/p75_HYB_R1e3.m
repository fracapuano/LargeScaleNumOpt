clear
close all force
clc
% shutting down the warning related to the tolerance for the
% gmres resolution in the Inexact NM
warning("off") 

f = @(x) p75_function(x); 
gradf = @(x) p75_gradient(x); 
Hessf = @(x) p75_hessian(x); 

m_improv = 10; 
eta_thresh = 1e2; 
thresh_improv = 5e-2; 

kmax = 1e3; 
tolgrad = 1e-6;

%R1000
n = 3;
dim = 10^n; 

% assembling the starting point
% x0 = [-1.2*ones(dim-1,1); -1]; %from here there is no convergence

x0 = ones(dim, 1); 
i = 1; 
while i <= dim
    x0(i) = -1.2; 
    i = i + 2; 
end

% Inexact Newton Method
load forcing_terms.mat

%backtracking parameters
btmax = 50; 
c1 = 1e-4; 
rho = 1e-1;

%inexact resolution parameters
gmres_maxit = 100; 
perc = 0.4; 

tic
[xk, fk, k, k_start, grads_out, values_out, inner_INN] = hybrid_PKplus_Newton(...
    x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax, fterms_quad, gmres_maxit, perc);
toc

%% Accessing to informations related to the number of inner iterations

% At first I define a number of bins
n_bins = 5; 

% The I return the histogram of the number of iterations
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

counts = histcounts(inner_INN,n_bins); 

tick_material = sort(counts); 

% "cutting" only top three ticks
tick_material = tick_material(3:end); 

figure(1)

bar(1:n_bins, counts)
title("{\bf Histogram of Inner Iterations}", "Interpreter", "latex")

% this changes when n_bins changes
xticklabels(["1:20", "20:40", "40:60", "60:80", "80:100"])

yticks(tick_material)
yticklabels(tick_material)
grid()