clear; clc; close all force

load METHODS_OUTCOME.mat
figure("Name", "Gradients and Objective values over iterations")

hold on
disc(1) = plot(grads_INN_disc, "LineWidth", 1.0); 
disc(2) = plot(grads_INN_disc_corrected, "LineWidth", 1.0); 
disc(3) = plot(grads_PK_disc, "LineWidth", 1.0); 
disc(4) = plot(grads_SD_disc, "LineWidth", 1.0); 

hold off
leg2 = legend("Inexact Newton Method", "INM with Hessian correction", "PK+", "Steepest Descent","Location","southwest");
set(leg2, "Interpreter", "latex")
set(gca, "YScale", "log")
set(gca, "XScale", "log")
set(gca,'TickLabelInterpreter','latex')
title("Gradient Norm over iterations starting in $x_0$ = (-1.2, 1)", "Interpreter", "latex")



