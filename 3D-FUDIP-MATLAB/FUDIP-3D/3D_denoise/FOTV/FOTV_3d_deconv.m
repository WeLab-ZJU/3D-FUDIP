% oldest version of decouple denoising and deblur
% f = loadtiff("D:\code\Julia\ThreeDeconv.jl\example\datasetwithraw\raw\cell\raw\cell_fitc.tif");
f = loadtiff('E:\data\hlm\denonv_test2\denonv_test2_crop_WBdenonv.tif');
f = single(f);
f = f./max(f(:));
raw = f;

[sx,sy,sz]=size(f);
sizeg=[sx,sy,sz];

alpha = 1.2;
K = 15; 
% beta = [0.2,0.2,0.2];
paral1 = 0;
fidelity = 1;
iter_num = 20;
mu = 0.008;

gpu = 1;

wx = zeros(1,K,1);wx(K) = 1;
wy = zeros(K,1,1);wy(K) = 1;
wz = zeros(1,1,K);wz(K) = 1;

for i = 1:(K-1)
    wx(1,K-i,1) = (-1)^i*gamma(alpha+1)/gamma(alpha-i+1)/factorial(i);
    wy(K-i,1,1) = wx(1,K-i,1);
    wz(1,1,K-i) = wx(1,K-i,1);
end

xtx = defDtD(wx,sizeg);
yty = defDtD(wy,sizeg);
ztz = defDtD(wz,sizeg);

% eigDtD = beta(1).*xtx + beta(2).*yty + beta(3).*ztz;
eigDtD = mu.*(xtx + yty +ztz);
eigA = single((fidelity/mu) + paral1^2 + eigDtD);

clear xtx yty ztz eigDtD 

if gpu==1
    f=gpuArray(f);
    eigA =gpuArray(eigA);
    bx = gpuArray.zeros(sizeg,'single');
    by = bx;
    bz = bx;
    bl1 =bx;
else
    bx = zeros(sizeg,'single');
    by = bx;
    bz = bx;
    bl1 =bx;
end

g_update = (fidelity/mu)*f;

for iter = 1:iter_num

    g_update = psf2otf(g_update);
    if iter>1
        g = real(otf2psf(g_update./eigA));
    else
        g = real(otf2psf(g_update./(fidelity/mu)));
    end
    
    g_update = (fidelity/mu).*f;
    
    [Lx,bx]=iter_xFotv(g,bx,wx,mu,alpha);
    g_update = g_update+Lx;
    
    [Ly,by]=iter_yFotv(g,by,wy,mu,alpha);
    g_update = g_update+Ly;
    
    [Lz,bz]=iter_zFotv(g,bz,wz,mu,alpha);
    g_update = g_update+Lz;
   
    [Lsparse,bl1]=iter_sparse(g,bl1,mu,paral1);
    g_update = g_update+Lsparse;

    disp(['  iter ' num2str(iter) ' | ' num2str(iter_num) ', done! ']);
%     progressbar(iter/iteration);
end

g(g<0)=0;

clear bxx byy bzz bxz bxy byz bl1 f normlize g_update

res = gather(g);
subplot(121),imshow(raw(:,:,13),[])
subplot(122),imshow(res(:,:,13),[])

% saveastiff(raw,'E:\data\hlm\denonv_test\denonv_test_crop.tif');
% saveastiff(res,'E:\data\hlm\denonv_test\denonv_test_crop_denoise.tif');