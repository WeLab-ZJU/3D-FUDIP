%***************************************************************************
% GPU-resident variant of Sparse-SIM algorithm core
%***************************************************************************
% Derived from SparseHessian_core.m (WeisongZhao, v1.0.3).
% Key difference: when f is already a gpuArray the function skips the
% redundant gpuArray upload, and allocates all buffers on the same device
% as f. The algorithm, variable names, and iteration structure are kept
% identical to avoid any numerical behaviour change.
%***************************************************************************
function g=SparseHessian_core_gpu(f,fidelity,contiz,paral1,iteration,gpu,mu)

if nargin < 5 || isempty(iteration)
    iteration=100;
end
if nargin < 6 || isempty(gpu)
    gpu=1;
end
if nargin < 7 || isempty(mu)
    mu=1;
end

% Determine effective device: honour both the explicit flag and the
% actual storage class of the input.
on_gpu = gpu || isa(f,'gpuArray');

contiz=single(sqrt(contiz));
f_flag=size(f,3);
if f_flag<3
    contiz=0;
    f(:,:,end+1:end+(3-size(f,3)))=repmat(f(:,:,end),[1,1,3-size(f,3)]);
    disp('Number of data frame is smaller than 3, the t or z-axis of continuity was turned off (conti=0)');
end
f=f./max(f(:));
f=single(f);
[sx,sy,sz]=size(f);
sizeg=[sx,sy,sz];

% Frequency-domain operator coefficients (computed on CPU, small arrays)
xxfft=operation_xx(sizeg);
yyfft=operation_yy(sizeg);
zzfft=operation_zz(sizeg);
xyfft=operation_xy(sizeg);
xzfft=operation_xz(sizeg);
yzfft=operation_yz(sizeg);
operationfft=1*xxfft+ 1*yyfft+(contiz^2)*zzfft+ 2*xyfft+ 2*(contiz)*xzfft+2*(contiz)*yzfft;
normlize = single((fidelity/mu) +paral1^2 +operationfft);
clear xxfft yyfft zzfft xyfft xzfft yzfft operationfft

if on_gpu
    % Upload only if not already on GPU
    if ~isa(f,'gpuArray')
        f=gpuArray(f);
    end
    if ~isa(normlize,'gpuArray')
        normlize=gpuArray(normlize);
    end
    bxx = gpuArray.zeros(sizeg,'single');
    byy = bxx;
    bzz = bxx;
    bxy =bxx;
    bxz =bxx;
    byz = bxx;
    bl1 =bxx;
else
    bxx = zeros(sizeg,'single');
    byy = bxx;
    bzz = bxx;
    bxy = bxx;
    bxz = bxx;
    byz = bxx;
    bl1 = bxx;
end

% Pass the effective device flag to iter_* helpers
gpu_flag = on_gpu;

g_update = (fidelity/mu)*f;
for iter = 1:iteration
    g_update = fftn(g_update);
    if iter>1
        g = real(ifftn(g_update./normlize));
    else
        g = real(ifftn(g_update./(fidelity/mu)));
    end
    g_update = (fidelity/mu)*f;
    
    [Lxx,bxx]=iter_xx(g,bxx,1,gpu_flag);
    g_update = g_update+Lxx;
    
    [Lyy,byy]=iter_yy(g,byy,1,gpu_flag);
    g_update = g_update+Lyy;
    
    [Lzz,bzz]=iter_zz(g,bzz,contiz^2,gpu_flag);
    g_update = g_update+Lzz;
    
    [Lxy,bxy]=iter_xy(g,bxy,2,gpu_flag);
    g_update = g_update+Lxy;
    
    [Lxz,bxz]=iter_xz(g,bxz,2*contiz,gpu_flag);
    g_update = g_update+Lxz;
    
    [Lyz,byz]=iter_yz(g,byz,2*contiz,gpu_flag);
    g_update = g_update+Lyz;
    
    [Lsparse,bl1]=iter_sparse(g,bl1,paral1,gpu_flag);
    g_update = g_update+Lsparse;
end
g(g<0)=0;
if f_flag<3
    g=g(:,:,2);
end
clear bxx byy bzz bxz bxy byz bl1 f normlize g_update
