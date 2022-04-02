function Hessian = p75_hessian(x)
% P75_HESSIAN returns the hessian of the function defined in Problem 75
% computed in the given point x

n = length(x); 
up_sub_diagonal = zeros(n,1);
main_diagonal = zeros(n,1); 

up_sub_diagonal(1) = -50*12*((x(2)-x(1))^2);
main_diagonal(1) = 1 + 50*12*((x(2)-x(1))^2);

for k=2:n-1

    up_sub_diagonal(k) = -50*12*((k)^2)*(x(k)-x(k+1))^2;
    main_diagonal(k) = 12*50*((k-1)^2)*((x(k-1)-x(k))^2) + 12*50*(k^2)*((x(k)-x(k+1))^2);
end

main_diagonal(n) = 50*12*((n-1)^2)*(x(n-1)-x(n))^2; 

Hessian = spdiags([up_sub_diagonal main_diagonal], [-1 0], n, n) + ...
          spdiags(up_sub_diagonal, -1, n, n)'; 

end

