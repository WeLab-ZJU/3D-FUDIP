function [Ls,bs] = iter_sparse(g,b,mu,lamda)

% update d
d = shrink_operator(g,b,mu);
% update b
bs = b + (g - d);
Ls = lamda.*(d-bs);

