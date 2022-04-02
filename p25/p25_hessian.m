function Hessf = p25_hessian(x)
%P25_HESSIAN returns the exact the Hessian for problem 25. 

n = length(x);
main_diag = zeros(n,1); 
up_sub_diag = zeros(n,1); 

for i=1:n
    if mod(i,2) == 1 %i is odd

        main_diag(i) = 600*x(i).^2 - 200.*x(i+1) + 1;
        up_sub_diag(i) = -200.*x(i); 

    else %i is even
        main_diag(i) = 100; 
    end
end

Hessf = spdiags([up_sub_diag, main_diag], [-1 0], n, n) + ...
       (spdiags(up_sub_diag, -1, n, n))'; 
end