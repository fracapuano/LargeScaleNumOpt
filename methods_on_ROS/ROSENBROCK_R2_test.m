clear
close all force
clc
%shutting down the warning related to the tolerance for the gmres resolution in the Inexact NM
warning("off") 
% defining the R2 rosenbrock function
f_ros = @(x)100*(x(2,:)-x(1,:).^2).^2+(1-x(1,:)).^2;

%defining the R2 gradient of the rosenbrock function
grad_ros = @(x) [...
    400*x(1,:).^3-400*x(1,:).*x(2,:)+2*x(1,:)-2; 200*(x(2,:)-x(1,:).^2)
    ]; 

%defining the R2 hessian of the rosenbrock function
hess_ros = @(x) [...
    1200*x(1, :)^2-400*x(2, :)+2, -400*x(1, :);
    -400*x(1, :), 200
    ];

% Inexact Newton Method
%%%%% TESTING INEXACT NEWTON METHOD %%%%%
load forcing_terms.mat

tol = 1e-9; 
bt_max = 50; 
c1 = 1e-4; 
btmax = 50;
rho = 1e-1;
gmres_maxit = 2; 
kmax = 1000;

%con ~ segni del punto iniziale concordi
x0_con = [1.2; 1.2]; 
tic

[~, fk, gradfk_norm, k_Inn1, grads_INN_con, values_INN_con, its_con] = ...
    innewton_general(x0_con, f_ros, grad_ros, hess_ros, kmax, tol, c1, rho, btmax, fterms_quad, gmres_maxit);

comp_time = toc;

disp("INEXACT NEWTON METHOD:")
disp("********************")
disp("STARTING FROM X0_CON")
disp("********************")
disp("*** FINISHED ***")
disp("-----")
fprintf("In %d/%d iterations a point in" + ...
    " which the norm of the gradient is" + ...
    " equal to %e has been found\n", k_Inn1, kmax, gradfk_norm)
disp("-----")
fprintf("The objective function in this point assumes the value: %d\n", fk)
disp("-----")
fprintf("\nThis solution has been obtained in %e seconds\n", comp_time)
disp("*** *** ***")
% % starting from x0_con convergence is reached in the first 100 iterations

% %disc ~ segni del punto iniziale discordi
x0_disc = [-1.2; 1]; 
tic

[~, fk, gradfk_norm, k_Inn2, grads_INN_disc, values_INN_disc, its_disc] = ...
    innewton_general(x0_disc, f_ros, grad_ros, hess_ros, kmax, tol, c1, rho, btmax, fterms_quad, gmres_maxit);

comp_time = toc;

disp("INEXACT NEWTON METHOD:")
disp("********************")
disp("STARTING FROM X0_DISC")
disp("********************")
disp("*** FINISHED ***")
disp("-----")
fprintf("In %d/%d iterations a point in" + ...
    " which the norm of the gradient is" + ...
    " equal to %e has been found\n", k_Inn2, kmax, gradfk_norm)
disp("-----")
fprintf("The objective function in this point assumes the value: %d\n", fk)
disp("-----")
fprintf("\nThis solution has been obtained in %e seconds\n", comp_time)
disp("*** *** ***")

% starting from x0_disc convergence is not reached at all

% PK+ with backtracking

%%%%% TESTING NON LINEAR POLAK RIBIERE "+" METHOD %%%%%

% starting in x0_con
tic
[~, fk, k_PR1, grads_PK_con, values_PK_con] = nonlinear_PKplus(...
    f_ros, grad_ros, tol, x0_con, kmax, c1, rho, btmax); 

comp_time = toc;
disp("PK+ NCGM:")
disp("********************")
disp("STARTING FROM X0_CON")
disp("********************")
disp("*** FINISHED ***")
disp("-----")
fprintf("In %d/%d iterations a point in" + ...
    " which the norm of the gradient is" + ...
    " equal to %e has been found\n", k_PR1, kmax, grads_PK_con(end))
disp("-----")
fprintf("The objective function in this point assumes the value: %d\n", fk)
disp("-----")
fprintf("\nThis solution has been obtained in %e seconds\n", comp_time)
disp("*** *** ***")
% converging
disp("********************")
disp("STARTING FROM X0_DISC")
disp("********************")
%starting in x0_disc
tic
[~, fk, k_PR2, grads_PK_disc, values_PK_disc] = nonlinear_PKplus(...
    f_ros, grad_ros, tol, x0_disc, kmax, c1, rho, btmax);
comp_time = toc;

disp("*** FINISHED ***")
disp("-----")
fprintf("In %d/%d iterations a point in" + ...
    " which the norm of the gradient is" + ...
    " equal to %e has been found\n", k_PR2, kmax, grads_PK_disc(end))
disp("-----")
fprintf("The objective function in this point assumes the value: %d\n", fk)
disp("-----")
fprintf("\nThis solution has been obtained in %e seconds\n", comp_time)
disp("*** *** ***")
%converging a little bit slower

% Steepest Descent with backtracking
%%%%% TESTING SD WITH BACKTRACKING %%%%%
alpha0 = 1; 
c1 = 1e-4; 
rho = 1e-1;
btmax = 100; 
tic
[~, fk, k_SD1, grads_SD_con, values_SD_con] = SD_backtrack(x0_con, f_ros, grad_ros, alpha0, ...
    kmax, tol, c1, rho, btmax); 
comp_time = toc;
disp("STEEPEST DESCENT")
disp("********************")
disp("STARTING FROM X0_CON")
disp("********************")
disp("*** FINISHED ***")
disp("-----")
fprintf("In %d/%d iterations a point in" + ...
    " which the norm of the gradient is" + ...
    " equal to %e has been found\n", k_SD1, kmax, grads_SD_con(end))
disp("-----")
fprintf("The objective function in this point assumes the value: %d\n", fk)
disp("-----")
fprintf("\nThis solution has been obtained in %e seconds\n", comp_time)
disp("*** *** ***")

tic
[~, fk, k_SD2, grads_SD_disc, values_SD_disc] = SD_backtrack(x0_disc, f_ros, grad_ros, alpha0, ...
    kmax, tol, c1, rho, btmax); 
comp_time = toc;
disp("********************")
disp("STARTING FROM X0_DISC")
disp("********************")
disp("*** FINISHED ***")
disp("-----")
fprintf("In %d/%d iterations a point in" + ...
    " which the norm of the gradient is" + ...
    " equal to %e has been found\n", k_SD2, kmax, grads_SD_disc(end))
disp("-----")
fprintf("The objective function in this point assumes the value: %d\n", fk)
disp("-----")
fprintf("\nThis solution has been obtained in %e seconds\n", comp_time)
disp("*** *** ***")

%converging, pretty fast too, but it is requiring kinda big number of
%iterations. 

%INNEWTON FAILED FROM X0_DISC, it is necessary to try something different
%like introducing correction

% %disc ~ segni del punto iniziale discordi

tic
[~, fk, gradfk_norm, k_InnCorr2, ~, grads_INN_disc_corrected, values_INN_disc_corrected] = ...
    innewton_general_with_correction(x0_disc, f_ros, grad_ros, hess_ros,...
                                     kmax, tol, c1, rho, btmax, fterms_quad, gmres_maxit);
comp_time = toc;

disp("INEXACT NEWTON METHOD WITH CORRECTION:")
disp("********************")
disp("STARTING FROM X0_DISC")
disp("********************")
disp("*** FINISHED ***")
disp("-----")
fprintf("In %d/%d iterations a point in" + ...
    " which the norm of the gradient is" + ...
    " equal to %e has been found\n", k_InnCorr2, kmax, gradfk_norm)
disp("-----")
fprintf("The objective function in this point assumes the value: %d\n", fk)
disp("-----")
fprintf("\nThis solution has been obtained in %e seconds\n", comp_time)
disp("*** *** ***")

save METHODS_OUTCOME
