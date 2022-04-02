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

kmax = 1e3; 
tol = 1e-6; 

%R1000
n = 3;
dim = 10^n;

%assembling the starting point
x0 = -1*ones(dim,1);

% Inexact Newton Method
load forcing_terms.mat

%backtracking parameters
btmax = 50; 
c1 = 1e-4; 
rho = 0.8;

%Newton conditions parameter
perc = 0.4; 

%inexact resolution parameters
gmres_maxit = 100; 
tic
[xk, fk, k, grads_1, values_1] = hybrid_PKplus_Newton(x0, f, gradf, hessf, kmax, ...
    tol, c1, rho, btmax, fterms_suplin, gmres_maxit, perc);
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

%%
%InNewton converges only in the case in which correction is performed

%%PK PLUS

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

% PK+ converges

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

% Regular steepest descent do not converge in the given number of
% iterations. Needs to have more iterations. 

%% Plots
close all force
figure(1)

sgtitle("Results in \bf R1e3 starting in $\bar{x}_0$", "Interpreter", "latex")

subplot(2,1,1)
hold on
plot(gradients_INN, "LineWidth", 1.0)
plot(gradients_cgm, "LineWidth", 1.0)
plot(grad_seq, "LineWidth", 1.0)

l = legend("Inexact Newton", "PK+ CGM", "Steepest Descent", "Location", "southwest");
set(l, "Interpreter", "latex")

title("\bf Gradient norm over iterations", "Interpreter", "latex")

k_stops = sort([k_INN, k_cgm, k_SD]); 
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

k_stops = sort([k_INN, k_cgm, k_SD]); 
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
