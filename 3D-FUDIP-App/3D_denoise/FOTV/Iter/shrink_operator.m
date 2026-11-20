function signd = shrink_operator(g,b,lamda)
signd = abs(g+b)-1./lamda;
signd(signd<0)=0;
signd = signd.*sign(g+b);
% signd = max(abs(g+b)-1./lamda, 0).*sign(g+b);
end