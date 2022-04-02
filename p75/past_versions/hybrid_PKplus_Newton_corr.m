function [xk, fk, k, grads_out, values_out] = hybrid_PKplus_Newton_corr(...
    x0, f, gradf, Hessf, kmax, tolgrad, c1, rho, btmax, fterms, pcg_maxit, m_improv, ...
    eta_thresh, thresh_improv)
%HYBRID METHOD THAT COMBINES PK+ AND NEWTON

xk = x0; 
gradf_xk = gradf(xk); 
alpha_0 = 1; 
perc = 0.80; 

% Function handle for the armijo condition
farmijo = @(fk, alpha, gradfk, pk) ...
    fk + c1 * alpha * gradfk' * pk;

alpha_k = alpha_0;
SD_Old = -gradf_xk;
p0 = SD_Old;
k = 1; 

grads = zeros(1, kmax); 
values = zeros(1, kmax); 
NewGradNorm = norm(gradf_xk);

pk = p0;
while k<kmax && NewGradNorm > tolgrad
    
    gradfk = gradf(xk);
    
    fk = f(xk);
    grads(k) = norm(gradfk);
    values(k) = fk;

    SD_New = -gradfk; 
    betaPR_k = (SD_New'*(SD_New-SD_Old))/(SD_Old'*SD_Old);

    %implementing the '+' variation
    betaPR_k = max(betaPR_k, 0); 

    pk = SD_New + betaPR_k*pk;
    alpha_k = alpha_0;
    %candidate new point
    xnew = xk + alpha_k * pk;
    fnew = f(xnew);
    
    bt = 0;
    % Backtracking strategy
    while bt < btmax && fnew > farmijo(fk, alpha_k, gradfk, pk)
        % Reduce the value of alpha
        alpha_k = rho * alpha_k;
        % recompute the new point
        xnew = xk + alpha_k * pk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
    end
    
    xk = xk + alpha_k*pk;

    SD_Old = SD_New;
    NewGradNorm = norm(SD_New)
    k = k + 1;
    
    if NewGradNorm < eta_thresh * tolgrad || k > floor(perc * kmax)
        if ((grads(k) - grads(k-m_improv))/(grads(k-m_improv))) < thresh_improv
        % switching to Inexact Newton Method
            break
        end
    end
end

k_start = k - 1; %Newton starts at the last k of PK+

[xk, fk, ~, k_N, grads_N, values_N] = h_innewton_general_with_correction(xk, f, gradf, Hessf, kmax-k_start, ...
    tolgrad, c1, rho, btmax, fterms, pcg_maxit);

k = k_start + k_N;

grads_1 = grads(1:k_start); 
values_1 = values(1:k_start); 

grads_out = [grads_1, grads_N]; 
values_out = [values_1, values_N]; 

k = k_start + k_N; 
end


    
