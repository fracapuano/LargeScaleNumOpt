function gradient = p75_gradient(x)
% P77_GRADIENT returns the analytical gradient of the function defined in
% problem 77.
n = length(x);
gradient = zeros(n, 1); 
gradient(1) = x(1) - 1 + 200*(x(1)-x(2))^3;

for k=2:n-1
    gradient(k) = 200*(k^2)*(x(k)-x(k+1))^3 - 200*((k-1)^2)*(x(k-1)-x(k))^3;
end

gradient(n) = -200*((n-1)^2)*((x(n-1)-x(n))^3);
end