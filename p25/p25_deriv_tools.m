%pattern emerges with at least n = 6

clear 
clc

syms x1 x2 x3 x4 x5 x6
f25 = 0.5*((10*(x1^2 - x2))^2 + (x1-1)^2 + (10*(x3^2 - x4))^2 + (x3 - 1)^2 + (10*(x5^2 - x6))^2 + (x5 - 1)^2);  

g25 = gradient(f25);
h25 = hessian(f25); 