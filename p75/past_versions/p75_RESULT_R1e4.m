clear
close all force
clc
%shutting down the warning related to the tolerance for the gmres resolution in the Inexact NM
warning("off") 

f_75 = @(x) p75_function(x); 
gradf_75 = @(x) p75_gradient(x); 
hessf_75 = @(x) p75_hessian(x); 

kmax = 2e3; 
tol = 1e-6;

%R10000
n = 4;
dim = 10^n; 

% assembling the starting point
% x0 = ones(dim,1);
% i = 1; 
% while i<=dim
%     x0(i) = 0; 
%     i = i + 2; 
% end

x0 = [-1.2*ones(dim-1,1); -1]; 

% Inexact Newton Method
load forcing_terms.mat

%backtracking parameters
btmax = 50; 
c1 = 1e-4; 
rho = 1e-1;

%% Inexact Newton Method
%inexact resolution parameters
gmres_maxit = 100; 

tic
[~, fINN, gradINN, k_INN, gradients_INN, fINN_seq] = innewton_general(x0, f_75, gradf_75, hessf_75, kmax, ...
    tol, c1, rho, btmax, fterms_quad, gmres_maxit);
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

%% Inexact Newton with Hessian modification
tic
[~, fk, ~, k_INN_corr, ~, grads_corr, values_corr] = innewton_general_with_correction(x0, f_75, gradf_75, hessf_75, kmax, ...
    tol, c1, rho, btmax, fterms_quad, gmres_maxit);
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

%%
%InNewton converges faster when correction is not performed

hold on
plot(gradients_INN)
plot(grads_corr)
set(gca, "XScale", "log", "YScale", "log")


%% PK PLUS

rho = 1e-2;
btmax = 100; 

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

% PK+ is not converging in the given number of iterations. However, since
% the PK+ is fairly fast it would make sense to increase the number of
% iterations to further obtain information with respect to convergence.

rho = 0.2; 
alpha0 = 1;
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

% Regular steepest descent do not converge in the given number of
% iterations. Needs to have more iterations. 

% % Plots

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
