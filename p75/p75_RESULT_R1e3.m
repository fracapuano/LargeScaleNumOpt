%clear
close all force
clc
%shutting down the warning related to the tolerance for the gmres resolution in the Inexact NM
warning("off") 

f_75 = @(x) p75_function(x); 
gradf_75 = @(x) p75_gradient(x); 
hessf_75 = @(x) p75_hessian(x); 

fterms_lin = @(gradf_norm) 0.5; 
fterms_suplin = @(gradf_norm) min([0.5, sqrt(gradf_norm)]); 
fterms_quad = @(gradf_norm) min([0.5, gradf_norm]); 

kmax = 1e3; 
tol = 1e-6;

%R1000
n = 3;
dim = 10^n; 

% assembling the starting point - Note that from the suggested starting
% poit convergence was not reached, hence I have experimented with an
% another one. 

x0 = ones(dim, 1); 
i = 1; 
while i <= dim
    x0(i) = -1.2; 
    i = i + 2; 
end

%backtracking parameters
btmax = 50; 
c1 = 1e-4; 
rho = 1e-1;

%% Inexact Newton Method
%inexact resolution parameters
gmres_maxit = 100;
restart = floor(sqrt(n/2)); 

tic
[~, fINN, ~, k_INN, gradients_INN, fINN_seq, inner_INN] = ...
    innewton_general(x0, f_75, gradf_75, hessf_75, kmax, ...
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

%% Inexact Newton Method with Hessian Correction
tic
max_corrections = 50; 
restart = []; 

[~, ~, ~, k_INN_corr, ~, grads_corr, values_corr, inner_its] = ...
    innewton_general_with_correction(x0, f_75, gradf_75, hessf_75, max_corrections, ...
    kmax, tol, c1, rho, btmax, fterms_quad, gmres_maxit, restart);
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
rho = 1e-1;

tic
[~, fk_cgm, k_cgm, gradients_cgm, values_cgm] = nonlinear_PKplus(...
    f_75, gradf_75, tol, x0, kmax, c1, rho, btmax);
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
rho = 0.2; 
alpha0 = 0.85;
btmax = 100; 

tic
[~, fk_SD, k_SD, grad_seq, fseq_SD] = SD_backtrack(x0, f_75, gradf_75, alpha0, ...
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

%% Hybrid Method
perc = 0.4; 
restart = floor(sqrt(n)); 
tic
[xk, fk, k, k_start, grads_out, values_out, inner_INN] = ...
    hybrid_PKplus_Newton(x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax, ...
                         fterms_quad, gmres_maxit, restart, perc);
comp_time = toc; 

fprintf("\nCOMPUTATIONAL TIME: %e seconds \n",comp_time)
disp("****************")
disp("HYBRID METHOD")
disp("****************")
disp("-----")
fprintf("In %d/%d its a solution with gradient" + ...
    " equal to %e has been found\n", k, kmax, grads_out(end))
disp("-----")

fprintf("In this point the function has value equal to %d\n", values_out(end))

disp("-----")
fprintf("This solutions has been found in %e seconds\n", comp_time)
disp("****************")

%% Plots

figure(1)

sgtitle("Results in \bf R1e3", "Interpreter", "latex")

subplot(2,1,1)
hold on
plot(gradients_INN, "LineWidth", 1.0)
plot(gradients_cgm, "LineWidth", 1.0)
plot(grad_seq, "LineWidth", 1.0)

l = legend("Inexact Newton", "PK+ CGM", "Steepest Descent", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Gradient norm over iterations", "Interpreter", "latex")

k_stops = sort([k_INN, k_cgm]); 
grad_stops = [gradients_INN(end), grad_seq(end)];

yticks(grad_stops)
ytickformat("%.3e")
xticks(k_stops)

xlabel("Number of iterations", "Interpreter", "latex")
ylabel("Norm of the gradient", "Interpreter", "latex")

set(gca, "TickLabelInterpreter", "latex")
set(gca, "YScale", "log")
set(gca, "XScale", "log")
set(gca, "YTickLabelMode", "auto")
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

k_stops = sort([k_INN, k_cgm]); 
f_stop = [fINN_seq(end), fseq_SD(end)];

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

%% Plots related to Hybrid Method vs. INM comparison

figure(1)

sgtitle("Results in \bf R1e3", "Interpreter", "latex")

subplot(2,1,1)
hold on
plot(gradients_INN, "LineWidth", 1.0)
plot(grads_out, "LineWidth", 1.0)

l = legend("Inexact Newton", "Hybrid PK-Newton", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Gradient norm over iterations", "Interpreter", "latex")

k_stops = sort([k]); 
grad_stops = sort([grads_out(end)]);

yticks(grad_stops)
ytickformat("%.3e")
xticks(k_stops)

xlabel("Number of iterations", "Interpreter", "latex")
ylabel("Norm of the gradient", "Interpreter", "latex")

set(gca, "TickLabelInterpreter", "latex")
set(gca, "YScale", "log")
set(gca, "XScale", "log")
set(gca, "YTickLabelMode", "auto")
grid on
hold off

%%%%%%%%%

subplot(2,1,2)
hold on
plot(fINN_seq, "LineWidth", 1.0)
plot(values_out, "LineWidth", 1.0)

l = legend("Inexact Newton", "Hybrid PK-Newton", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Objective function value over iterations", "Interpreter", "latex")

k_stops = sort([k]); 
f_stop = sort([values_out(end)]);

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
