function out = FOTV_wraper(in,args)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = FOTV_wraper(in,sigma)
% performs FOTV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fidelity = args.fidelity;
mu = args.mu;
iter_num = args.iter_num;
gpu = args.gpu;
out = FOTV_denoise(in,fidelity,mu,iter_num,gpu);

end