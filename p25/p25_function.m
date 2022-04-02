function value = p25_function(x)
%P25_FUNCTION returns the value of the function defined in Problem 25

n = length(x); F = 0; 
for i=1:n
    if mod(i,2)==1
        F = F + (10*(x(i)^2 - x(i+1)))^2; 
    else
        F = F + ((x(i-1)-1)^2);
    end
end
value = F/2;
end
