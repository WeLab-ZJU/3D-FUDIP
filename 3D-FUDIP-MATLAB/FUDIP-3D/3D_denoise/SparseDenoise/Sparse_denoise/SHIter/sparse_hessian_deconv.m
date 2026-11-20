function g = sparse_hessian_deconv(f,psf,fidelity,contiz,paral1,iteration,gpu,mu)

addpath('D:\code\MATLAB\sparse_deconv_v0.3\SHOperation\')
if nargin < 6 || isempty(iteration)
    iteration=100;
end
if nargin < 7 || isempty(gpu)
    gpu=cudaAvailable;
end
if nargin < 8 || isempty(mu)
    mu=1;
end
% progressbar('Sparsity reconstruction');
contiz=single(sqrt(contiz));
f_flag=size(f,3);
if f_flag<3
    contiz=0;
    f(:,:,end+1:end+(3-size(f,3)))=repmat(f(:,:,end),[1,1,3-size(f,3)]);
    disp('Number of data frame is smaller than 3, the t or z-axis of continuity was turned off (conti=0)');
end

% f=single(f);
% f=f./max(f(:));

[sx,sy,sz]=size(f);
sizeg=[sx,sy,sz];

psf = gpuArray(psf);
OTF = single(psf2otf(psf,sizeg));
clear psf


xxfft=operation_xx(sizeg);
yyfft=operation_yy(sizeg);
zzfft=operation_zz(sizeg);
xyfft=operation_xy(sizeg);
xzfft=operation_xz(sizeg);
yzfft=operation_yz(sizeg);
operationfft=1*xxfft+ 1*yyfft+(contiz^2)*zzfft+ 2*xyfft+ 2*(contiz)*xzfft+2*(contiz)*yzfft;
normlize = single((fidelity/mu)*conj(OTF).*OTF +paral1^2 +operationfft);
clear xxfft yyfft zzfft xyfft xzfft yzfft operationfft
if gpu==1
    f=gpuArray(f);
    normlize =gpuArray(normlize);
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
g_update = (fidelity/mu).*f;
imshow(g_update(:,:,10),[])
for iter = 1:iteration
    tic;
    g_update = psf2otf(g_update);
    if iter>1
        g = real(otf2psf(g_update./normlize));
    else
        g = real(otf2psf(g_update./(fidelity/mu)));
    end
    g_update = real((fidelity/mu)*otf2psf(conj(OTF).*psf2otf(g)));
    imshow(g(:,:,10),[])
    [Lxx,bxx]=iter_xx(g,bxx,1,gpu);
    g_update = g_update+Lxx;
    
    [Lyy,byy]=iter_yy(g,byy,1,gpu);
    g_update = g_update+Lyy;
    
    [Lzz,bzz]=iter_zz(g,bzz,contiz^2,gpu);
    g_update = g_update+Lzz;
    
    [Lxy,bxy]=iter_xy(g,bxy,2,gpu);
    g_update = g_update+Lxy;
    
    [Lxz,bxz]=iter_xz(g,bxz,2*contiz,gpu);
    g_update = g_update+Lxz;
    
    [Lyz,byz]=iter_yz(g,byz,2*contiz,gpu);
    g_update = g_update+Lyz;
    
    [Lsparse,bl1]=iter_sparse(g,bl1,paral1,gpu);
    g_update = g_update+Lsparse;
    ttime = toc;
    disp(['  iter ' num2str(iter) ' | ' num2str(iteration) ', took ' num2str(ttime) ' secs']);
%     progressbar(iter/iteration);
end
%g(g<0)=0;
if f_flag<3
    g=g(:,:,2);
end
% g = g(5:716,20:691,52:155);
clear bxx byy bzz bxz bxy byz bl1 f normlize g_update
end