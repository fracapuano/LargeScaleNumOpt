function value = p75_function(x)
% P75_FUNCTION returns the value of the function defined in Problem 77

n = length(x); 
F = 0; 
f_1 = x(1)-1;
F = F + (f_1)^2;

for i=2:n
    F = F + (10*(i-1)*(x(i)-x(i-1))^2)^2;
end
value = F/2; 
end