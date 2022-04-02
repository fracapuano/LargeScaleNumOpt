function gradf = p25_gradient(x)
%P25_GRADIENT returns the exact analytical gradient of the function defined
%in Problem 25. 
n = length(x); 
gradf = zeros(n,1); 

for i=1:n
    if mod(i,2) == 1 %i is odd
        gradf(i) = 200*x(i).^3 - 200*x(i).*x(i+1) + x(i) - 1; 
    else %i is even
        gradf(i) = 100*(x(i) - (x(i-1).^2)); 
    end
end

