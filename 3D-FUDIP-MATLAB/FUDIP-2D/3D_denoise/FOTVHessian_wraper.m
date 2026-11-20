function out = FOTVHessian_wraper(in,args)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = FOTV_wraper(in,sigma)
% performs FOTV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fidelity = args.fidelity;
mu = args.mu;
iter_num = args.iter_num;
contiz = args.contiz;
gpu = args.gpu;
paral1 = 0;

% out = FOTV_denoise(in,fidelity,mu,iter_num,gpu);
out = FOTV_Hessian_denoise(in,fidelity,mu,paral1,iter_num,contiz,gpu);

end