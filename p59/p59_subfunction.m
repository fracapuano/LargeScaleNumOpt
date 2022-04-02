function f_value = p59_subfunction(x, j)
%P59_SUBFUNCTION returns the f_value corresponding to position index j for
%the given input x for the function defined in Problem 59. 

n = length(x); 
switch j
    case 1
        f_value = x(j)*(0.5*x(j) - 3) + 2*x(j+1) - 1; 
    case n
        f_value = x(j)*(0.5*x(j) - 3) - 1 + x(j-1); 
    otherwise
        f_value = x(j)*(0.5*x(j) - 3) + x(j-1) + 2*x(j+1) - 1; 
end

