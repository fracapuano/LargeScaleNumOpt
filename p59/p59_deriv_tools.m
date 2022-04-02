%pattern emerges with at least n = 6

clear 
clc

% syms x1 x2 x3 x4 x5 x6 x7 x8 x9 x_10
% f59 = 0.5*( ...
%       (x1*(0.5*x1 - 3) + 2*x2 - 1)^2 + ...
%       (x2*(0.5*x2 - 3) + x1 + 2*x3 - 1)^2 + ...
%       (x3*(0.5*x3 - 3) + x2 + 2*x4 - 1)^2 + ...
%       (x4*(0.5*x4 - 3) + x3 + 2*x5 - 1)^2 + ...
%       (x5*(0.5*x5 - 3) + x4 + 2*x6 - 1)^2 + ...
%       (x6*(0.5*x6 - 3) + x5 + 2*x7 - 1)^2 + ...
%       (x7*(0.5*x7 - 3) + x6 + 2*x8 - 1)^2 + ...
%       (x8*(0.5*x8 - 3) + x7 + 2*x9 - 1)^2 + ...
%       (x9*(0.5*x9 - 3) + x8 + 2*x_10 - 1)^2 + ...
%       (x_10*(0.5*x_10 - 3) - 1 + x9)^2 ...
%       );

syms x1 x2 x3 x4 x5 x6
f59 = 0.5*( ...
      (x1*(0.5*x1 - 3) + 2*x2 - 1)^2 + ...
      (x2*(0.5*x2 - 3) + x1 + 2*x3 - 1)^2 + ...
      (x3*(0.5*x3 - 3) + x2 + 2*x4 - 1)^2 + ...
      (x4*(0.5*x4 - 3) + x3 + 2*x5 - 1)^2 + ...
      (x5*(0.5*x5 - 3) + x4 + 2*x6 - 1)^2 + ...
      (x6*(0.5*x6 - 3) - 1 + x5)^2 ...
      );

g59 = gradient(f59);
h59 = hessian(f59); 

h59_handle = @(x) [
2*x(2) + x(1)*(x(1)/2 - 3) + (x(1) - 3)^2,        2*x(1) + x(2) - 9,      2,         0,         0,      0;
       2*x(1) + x(2) - 9,                 x(1) + 2*x(3) + x(2)*(x(2)/2 - 3) + (x(2) - 3)^2 + 4,                              2*x(2) + x(3) - 9,                                          2,                                          0,                                   0;
               2,                          2*x(2) + x(3) - 9,                               x(2) + 2*x(4) + x(3)*(x(3)/2 - 3) + (x(3) - 3)^2 + 4,                              2*x(3) + x(4) - 9,                                    2,                                   0;
                                0,                                          2,                              2*x(3) + x(4) - 9, x(3) + 2*x(5) + x(4)*(x(4)/2 - 3) + (x(4) - 3)^2 + 4,                           2*x(4) + x(5) - 9,                         2;
                                0,                                          0,                                          2,                              2*x(4) + x(5) - 9, x(4) + 2*x(6) + x(5)*(x(5)/2 - 3) + (x(5) - 3)^2 + 4,                       2*x(5) + x(6) - 9;
                                0,                                          0,                                          0,                                          2,                              2*x(5) + x(6) - 9, x(5) + x(6)*(x(6)/2 - 3) + (x(6) - 3)^2 + 3
                  ];         


g59_handle = @(x) [
    x(1) + 2*x(3) + x(2)*(x(2)/2 - 3) + (x(1) - 3)*(2*x(2) + x(1)*(x(1)/2 - 3) - 1) - 1;
   5*x(2) + 2*x(4) + 2*x(1)*(x(1)/2 - 3) + x(3)*(x(3)/2 - 3) + (x(2) - 3)*(x(1) + 2*x(3) + x(2)*(x(2)/2 - 3) - 1) - 3;
2*x(1) + 5*x(3) + 2*x(5) + 2*x(2)*(x(2)/2 - 3) + x(4)*(x(4)/2 - 3) + (x(3) - 3)*(x(2) + 2*x(4) + x(3)*(x(3)/2 - 3) - 1) - 3;
2*x(2) + 5*x(4) + 2*x(6) + 2*x(3)*(x(3)/2 - 3) + x(5)*(x(5)/2 - 3) + (x(4) - 3)*(x(3) + 2*x(5) + x(4)*(x(4)/2 - 3) - 1) - 3;
   2*x(3) + 5*x(5) + 2*x(4)*(x(4)/2 - 3) + x(6)*(x(6)/2 - 3) + (x(5) - 3)*(x(4) + 2*x(6) + x(5)*(x(5)/2 - 3) - 1) - 3;
                          2*x(4) + 4*x(6) + 2*x(5)*(x(5)/2 - 3) + (x(6) - 3)*(x(5) + x(6)*(x(6)/2 - 3) - 1) - 2
                  ]; 

x = rand(6,1); 







