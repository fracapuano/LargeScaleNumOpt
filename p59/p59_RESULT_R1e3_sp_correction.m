clear
close all force
clc
%shutting down the warning related to the tolerance for the gmres resolution in the Inexact NM
warning("off") 

% parameters for numeric differentiation
h_grad = sqrt(eps); 
h_hess = sqrt(h_grad); 
type_diff = "fw"; 

f = @(x) p59_function(x); 
gradf = @(x) p59_gradient(x, h_grad, type_diff); 
hessf = @(x) p59_hessian(f, x, h_hess); 

fterms_lin = @(gradf_norm) 0.5; 
fterms_suplin = @(gradf_norm) min([0.5, sqrt(gradf_norm)]); 
fterms_quad = @(gradf_norm) min([0.5, gradf_norm]); 

kmax = 1e3; 
tol = 1e-6; 

%R1000
n = 3;
dim = 10^n; 

%assembling the starting point
% assigned starting point 
% x0 = -1*ones(dim,1);

% xbar_0
x0 = ones(dim,1); 

% backtracking parameters
btmax = 50; 
c1 = 1e-4; 
rho = 0.8;

%% Inexact Newton Method
%inexact resolution parameters
gmres_maxit = 100; 
restart = [];

tic
[x_fin, fINN, ~, k_INN, gradients_INN, fINN_seq, inner_INN] = 
    innewton_general(x0, f, gradf, hessf, kmax, ...
    tol, c1, rho, btmax, fterms_quad, gmres_maxit, restart);
comp_time = toc;

disp("****************")
disp("INEXACT NEWTON METHOD")
disp("****************")
disp("-----")
fprintf("In %d/%d its a solution with gradient" + ...
    " equal to %e has been found\n", k_INN, kmax, gradients_INN(end))
disp("-----")

fprintf("In this point the function has value equal to %d\n", fINN_seq(end))

disp("-----")
fprintf("This solutions has been found in %e seconds\n", comp_time)
disp("****************")

%% Accessing to informations related to the number of inner iterations

% At first I define a number of bins
n_bins = 5; 

% The I return the histogram of the number of iterations
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

counts = histcounts(inner_INN,n_bins); 

tick_material = sort(unique(counts)); 

figure(1)

bar(1:n_bins, counts)
title("{\bf Histogram of Inner Iterations}", "Interpreter", "latex")

% this changes when n_bins changes
xticklabels(["1:20", "20:40", "40:60", "60:80", "80:100"])

yticks(tick_material)
yticklabels(tick_material)
grid()

%% Inexact Newton Method with Hessian correction
max_corrections = 50; 
restart = [];

tic
[~, fk, ~, k_INN_corr, ~, grads_corr, values_corr] = 
    innewton_general_with_correction(x0, f, gradf, hessf, max_corrections, kmax, ...
    tol, c1, rho, btmax, fterms_quad, gmres_maxit, restart);
comp_time = toc;

disp("****************")
disp("INEXACT NEWTON METHOD WITH CORRECTION")
disp("****************")
disp("-----")
fprintf("In %d/%d its a solution with gradient" + ...
    " equal to %e has been found\n", k_INN_corr, kmax, grads_corr(end))
disp("-----")

fprintf("In this point the function has value equal to %d\n", values_corr(end))

disp("-----")
fprintf("This solutions has been found in %e seconds\n", comp_time)
disp("****************")

%% PK PLUS

rho = 0.3;
tic
[~, fk_cgm, k_cgm, gradients_cgm, values_cgm] = nonlinear_PKplus(...
    f, gradf, tol, x0, kmax, c1, rho, btmax);
comp_time = toc;

disp("****************")
disp("PK+ CONJ. GM")
disp("****************")
disp("-----")
fprintf("In %d/%d its a solution with gradient" + ...
    " equal to %e has been found\n", k_cgm, kmax, gradients_cgm(end))
disp("-----")

fprintf("In this point the function has value equal to %d\n", fk_cgm)

disp("-----")
fprintf("This solutions has been found in %e seconds\n", comp_time)
disp("****************")

%% Steepest Descent

rho = 0.3; 
alpha0 = 0.85;
btmax = 100; 

kmax = 1e3;

tic
[~, fk_SD, k_SD, grad_seq, fseq_SD] = SD_backtrack(x0, f, gradf, alpha0, ...
                                      kmax, tol, c1, rho, btmax);
comp_time = toc;

fprintf("\nCOMPUTATIONAL TIME: %e seconds \n",comp_time)
disp("****************")
disp("SIMPLE STEEPEST DESCENT")
disp("****************")
disp("-----")
fprintf("In %d/%d its a solution with gradient" + ...
    " equal to %e has been found\n", k_SD, kmax, grad_seq(k_SD))
disp("-----")

fprintf("In this point the function has value equal to %d\n", fseq_SD(k_SD))

disp("-----")
fprintf("This solutions has been found in %e seconds\n", comp_time)
disp("****************")

%% Plots related to the three methods
close all force
figure(1)

sgtitle("Results in {\bf R1e3}", "Interpreter", "latex")

subplot(2,1,1)
hold on
plot(gradients_INN, "LineWidth", 1.0)
plot(gradients_cgm, "LineWidth", 1.0)
plot(grad_seq, "LineWidth", 1.0)

l = legend("Inexact Newton", "PK+ CGM", "Steepest Descent", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Gradient norm over iterations", "Interpreter", "latex")

k_stops = sort([k_INN, k_SD]);
grad_stops = sort([gradients_INN(end)]);

yticks(grad_stops)
ytickformat("%.3e")
xticks(k_stops)

xlabel("Number of iterations", "Interpreter", "latex")
ylabel("Norm of the gradient", "Interpreter", "latex")

set(gca, "TickLabelInterpreter", "latex")
set(gca, "YScale", "log")
set(gca, "XScale", "log")
grid on
hold off

%%%%%%%%%

subplot(2,1,2)
hold on
plot(fINN_seq, "LineWidth", 1.0)
plot(values_cgm, "LineWidth", 1.0)
plot(fseq_SD, "LineWidth", 1.0)

l = legend("Inexact Newton", "PK+ CGM", "Steepest Descent", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Objective function value over iterations", "Interpreter", "latex")

k_stops = sort([k_INN, k_SD]); 
f_stop = sort([values_cgm(end), fINN_seq(end)]);

yticks(f_stop)
ytickformat("%.3e")
xticks(k_stops)

xlabel("Number of iterations", "Interpreter", "latex")
ylabel("Objective function value", "Interpreter", "latex")

set(gca, "TickLabelInterpreter", "latex")
set(gca, "YScale", "log")
set(gca, "XScale", "log")
grid on
hold off
