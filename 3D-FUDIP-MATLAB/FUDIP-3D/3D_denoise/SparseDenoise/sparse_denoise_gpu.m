function SHVideo = sparse_denoise_gpu(inputImg,fidelity,sparsity,zconti,SHIter,gpu)
% GPU-resident variant of sparse_denoise.
% Preserves the device placement of inputImg (CPU or GPU) throughout
% preprocessing, and delegates to SparseHessian_core_gpu which also
% respects the input device. This eliminates per-iteration
% gather -> gpuArray round-trips when called from the main loop.

warning('off');

% Unified device flag: use GPU if explicitly requested OR if data is
% already on GPU.
use_gpu = gpu || isa(inputImg,'gpuArray');

Oversample = 3;

%%%%%%%%%%%%%%%%%%% Load Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SIMmovie=inputImg;

if size(SIMmovie,3)<3
    flage3=size(SIMmovie,3);
    SIMmovie=padarray(SIMmovie,[0,0,3-size(SIMmovie,3)],'replicate' );
else
    flage3=size(SIMmovie,3);
end

%%%%%%%%%%%% Preprocessing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SIMmovie(SIMmovie < 0)=0;

switch Oversample
    case 1
        y=Spatial_Oversample(SIMmovie);
        f=single(y);
    case 2
        y=Fourier_Oversample(SIMmovie);
        f=single(y);
    case 3
        f = SIMmovie;
end

%%%%%%%%%%%%% RUN Reconstruction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SHVideo=SparseHessian_core_gpu(f,fidelity,zconti,sparsity,SHIter,use_gpu);
SHVideo=SHVideo(:,:,1:flage3);
end
