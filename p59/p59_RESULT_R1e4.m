clear
close all force
clc
%shutting down the warning related to the tolerance for the gmres resolution in the Inexact NM
warning("off") 

% parameters for numeric differentiation
h_grad = 1e-6; 
h_hess = 1e-3; 
type_diff = "fw"; 

f = @(x) p59_function(x); 
gradf = @(x) p59_gradient(x, h_grad, type_diff); 
hessf = @(x) p59_hessian(f, x, h_hess); 

kmax = 1e3; 
tol = 1e-6; 

%R1e4
n = 4;
dim = 10^n; 

%assembling the starting point

%x0 = -1*ones(dim,1);
x0 = ones(dim,1);

% Inexact Newton Method
load forcing_terms.mat

%backtracking parameters
btmax = 50; 
c1 = 1e-4; 
rho = 0.8;

tic
tmp = hessf(x0); 
hess_comp_time = toc; 
disp("The computational time needed to approximate the Hessian is "+hess_comp_time + " seconds")

%% PK PLUS

rho = 0.3;
btmax = 100; 
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

%%
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

% Regular steepest descent do not converge in the given number of
% iterations. Needs to have more iterations. 
%% Hybrid method

perc = 0.09;
rho = 0.4; 
gmres_maxit = 100; 
tic
[xk, fk, k, grads_1, values_1] = ...
    hybrid_PKplus_Newton(x0, f, gradf, hessf, kmax, ...
    tol, c1, rho, btmax, fterms_quad, gmres_maxit, perc);
comp_time = toc; 
disp("****************")
disp("HYBRID METHOD")
disp("****************")
disp("-----")
fprintf("In %d/%d its a solution with gradient" + ...
    " equal to %e has been found\n", k, kmax, grads_1(end))
disp("-----")

fprintf("In this point the function has value equal to %d\n", values_1(end))

disp("-----")
fprintf("This solutions has been found in %e seconds\n", comp_time)
disp("****************")

%% Plots
close all force
figure(1)

sgtitle("Results in \bf R1e4", "Interpreter", "latex")

subplot(2,1,1)
hold on
plot(grads_1, "LineWidth", 1.0)
plot(gradients_cgm, "LineWidth", 1.0)
plot(grad_seq, "LineWidth", 1.0)

l = legend("Hybrid PK-Newton", "PK+", "Steepest Descent", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Gradient norm over iterations", "Interpreter", "latex")

k_stops = sort([k, k_cgm, k_SD]); 
grad_stops = sort([grads_1(end), gradients_cgm(end)]);

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
plot(values_1, "LineWidth", 1.0)
plot(values_cgm, "LineWidth", 1.0)
plot(fseq_SD, "LineWidth", 1.0)

l = legend("Hybrid PK-Newton", "PK+", "Steepest Descent", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Objective function value over iterations", "Interpreter", "latex")

k_stops = sort([k, k_cgm, k_SD]); 
f_stop = sort([values_1(end), values_cgm(end)]);

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
