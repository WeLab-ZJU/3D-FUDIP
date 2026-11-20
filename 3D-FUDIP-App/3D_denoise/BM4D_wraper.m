function out = BM4D_wraper(in,args)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% out = BM4D_wraper(in,args)
% performs BM4D denoising
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma = args.sigma;
out = BM4D_denoise(in,sigma);

end