function g = FOTV_Hessian_denoise(f,fidelity,mu,paral1,iter_num,contiz,gpu)

[sx,sy,sz]=size(f);
sizeg=[sx,sy,sz];

alpha = 1.2;
K = 15;

[w_x,w_y,w_z,w_xy,w_xz,w_yz] = def_filter(K,alpha);

xtx = defDtD(w_x,sizeg);
yty = defDtD(w_y,sizeg);
ztz = defDtD(w_z,sizeg);
xtz = defDtD(w_xz,sizeg);
ytz = defDtD(w_yz,sizeg);
xty = defDtD(w_xy,sizeg);

eigDtD = 1*xtx+ 1*yty+(contiz^2)*ztz+ 2*xty+ 2*(contiz)*xtz+2*(contiz)*ytz;
eigA = single((fidelity/mu) + paral1^2 + eigDtD);

clear xtx yty ztz xty xtz ytz eigDtD 

if gpu==1
    f=gpuArray(single(f));
    eigA =gpuArray(eigA);
    bx = gpuArray.zeros(sizeg,'single');
    by = bx;
    bz = bx;
    bxy =bx;
    bxz =bx;
    byz = bx;
    bl1 =bx;
else
    bx = zeros(sizeg,'single');
    by = bx;
    bz = bx;
    bxy =bx;
    bxz =bx;
    byz = bx;
    bl1 =bx;
end

g_update = single((fidelity/mu)*f);

for iter = 1:iter_num

    g_update = psf2otf(g_update);
    if iter>1
        g = real(otf2psf(g_update./eigA));
    else
        g = real(otf2psf(g_update./(fidelity/mu)));
    end
    
    g_update = (fidelity/mu).*f;
    
    [Lx,bx]=iter_xFotv(g,bx,w_x,mu,alpha);
    g_update = g_update+Lx;
    
    [Ly,by]=iter_yFotv(g,by,w_y,mu,alpha);
    g_update = g_update+Ly;
    
    [Lz,bz]=iter_zFotv(g,bz,w_z,mu,alpha);
    g_update = g_update+Lz;
    
    [Lxy,bxy]=iter_xy(g,bxy,2,gpu);
    g_update = g_update+Lxy;
    
    [Lxz,bxz]=iter_xz(g,bxz,2*contiz,gpu);
    g_update = g_update+Lxz;
    
    [Lyz,byz]=iter_yz(g,byz,2*contiz,gpu);
    g_update = g_update+Lyz;
   
    [Lsparse,bl1]=iter_sparse(g,bl1,mu,paral1);
    g_update = g_update+Lsparse;

%     disp(['  iter ' num2str(iter) ' | ' num2str(iter_num) ', done! ']);

end

g(g<0)=0;
clear bxx byy bzz bxz bxy byz bl1 f eigA g_update

end

function [w_x,w_y,w_z,w_xy,w_xz,w_yz] = def_filter(K,alpha)

w_x = zeros(1,K,1);w_x(K) = 1;
w_y = zeros(K,1,1);w_y(K) = 1;
w_z = zeros(1,1,K);w_z(K) = 1;

for i = 1:(K-1)
    w_x(1,K-i,1) = (-1)^i*gamma(alpha+1)/gamma(alpha-i+1)/factorial(i);
    w_y(K-i,1,1) = w_x(1,K-i,1);
    w_z(1,1,K-i) = w_x(1,K-i,1);
end

w_xy = [1 -1;-1 1];

w_xz(:,:,1)=[1,-1];
w_xz(:,:,2)=[-1,1];

w_yz(:,:,1)=[1;-1];
w_yz(:,:,2)=[-1;1];

w_x = gpuArray(double(w_x));
w_y = gpuArray(double(w_y));
w_z = gpuArray(double(w_z));
w_xz = gpuArray(double(w_xz));
w_yz = gpuArray(double(w_yz));
w_xy = gpuArray(double(w_xy));

end