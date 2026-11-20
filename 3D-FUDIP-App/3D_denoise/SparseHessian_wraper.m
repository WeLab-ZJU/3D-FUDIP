function out = SparseHessian_wraper(in,args)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = FOTV_wraper(in,sigma)
% performs FOTV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fidelity = args.fidelity;
sparsity = args.sparsity;
iter_num = args.iter_num;
gpu = args.gpu;
zconti = args.zconti;
out = sparse_denoise(in,fidelity,sparsity,zconti,iter_num,gpu);

end