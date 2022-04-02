clear
close all force
clc
% shutting down the warning related to the tolerance for the gmres 
% resolution in the Inexact NM
warning("off") 

f = @(x) p25_function(x); 
gradf = @(x) p25_gradient(x); 
hessf = @(x) p25_hessian(x); 

fterms_lin = @(gradf_norm) 0.5; 
fterms_suplin = @(gradf_norm) min([0.5, sqrt(gradf_norm)]); 
fterms_quad = @(gradf_norm) min([0.5, gradf_norm]); 

kmax = 1e3; 
tol = 1e-6; 

%R1e5
n = 5;
dim = 10^n; 

%assembling the starting point
x0 = ones(dim,1);
i = 1; 
while i<=dim
    x0(i) = -1.2; 
    i = i + 2; 
end

%% Inexact Newton Method

%backtracking parameters
btmax = 50; 
c1 = 1e-4; 
rho = 0.8;

%inexact resolution parameters
gmres_maxit = 100; 
restart = floor(sqrt(n)); 

tic
[~, fINN, ~, k_INN, gradients_INN, fINN_seq] = ...
    innewton_general(x0, f, gradf, hessf, kmax, tol, c1, rho, btmax, ...
    fterms_quad, gmres_maxit, restart);
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
%% Inexact Newton Method with Hessian Correction
tic
max_correctinos = 50; 
restart = []; 

[~, ~, ~, k_INN_corr, grads_corr, values_corr, its_corr] = ...
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

rho = 0.2;
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

%% Hybrid method

% parameters defined via observing the trajectory of the other methods.
perc = 0.18; 
rho = 0.2; 

tic
[xk, fk, k, k_start, grads_out, values_out] = hybrid_PKplus_Newton(...
    x0, f, gradf, hessf, kmax, tol, c1, rho, btmax, fterms_quad, gmres_maxit, perc); 
comp_time = toc;

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
%% Plots related to the three methods

figure(1)
sgtitle("Results in \bf R1e5", "Interpreter", "latex")

subplot(2,1,1)
hold on
plot(grads_corr, "LineWidth", 1.0)
plot(gradients_cgm, "LineWidth", 1.0)
plot(grad_seq, "LineWidth", 1.0)

l = legend("Inexact Newton", "PK+ CGM", "Steepest Descent", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Gradient norm over iterations", "Interpreter", "latex")

k_stops = sort([k_INN_corr, k_cgm]); 
grad_stops = sort([gradients_cgm(end), grads_corr(end)]);

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
plot(values_corr, "LineWidth", 1.0)
plot(values_cgm, "LineWidth", 1.0) 
plot(fseq_SD, "LineWidth", 1.0)

l = legend("PK+ CGM", "Steepest Descent", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Objective Function value over iterations", "Interpreter", "latex")

k_stops = sort([k_INN_corr, k_cgm]); 
f_stop = sort([values_cgm(end), values_corr(end)]);

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

%% Plots related to the PK+ vs. Hybrid PK-Newton comparison

figure(2)

subplot(2,1,1)
hold on
plot(grads_out, "LineWidth", 1.0, "Color", "black", "LineStyle", "--")
plot(gradients_cgm, "LineWidth", 1.0, "Color", "red")

l = legend("Hybrid PK-Newton", "PK+ CGM", "Location", "southwest");

set(l, "Interpreter", "latex")

title("\bf Gradient norm over iterations", "Interpreter", "latex")

k_stops = sort([k, k_cgm]); 
grad_stops = sort([grads_out(end)]);

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
plot(values_out, "LineWidth", 1.0, "Color", "black", "LineStyle", "--")
plot(values_cgm, "LineWidth", 1.0, "Color", "red")

l = legend("Hybrid PK-Newton", "PK+", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Objective Function value over iterations", "Interpreter", "latex")

k_stops = sort([k, k_cgm]); 
f_stop = sort([values_cgm(end), values_out(end)]);

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