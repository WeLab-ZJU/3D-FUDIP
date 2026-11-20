function out = SparseHessian_wraper_gpu(in,args)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GPU-resident variant of SparseHessian_wraper.
% Input `in` may already be a gpuArray; this wrapper avoids
% unnecessary gather/gpuArray round-trips by forwarding
% directly to the GPU-aware sparse_denoise_gpu.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fidelity = args.fidelity;
sparsity = args.sparsity;
iter_num = args.iter_num;
gpu = args.gpu;
zconti = args.zconti;
out = sparse_denoise_gpu(in,fidelity,sparsity,zconti,iter_num,gpu);

end
