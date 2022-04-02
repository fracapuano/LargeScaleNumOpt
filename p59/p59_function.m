function F = p59_function(x)
%P59_FUNCTION returns the value of the function defined in Problem 59

[n,~] = size(x); 
value = 0; 
for k=1:n
    value = value + (p59_subfunction(x, k))^2;
end
F = value/2; 
end

