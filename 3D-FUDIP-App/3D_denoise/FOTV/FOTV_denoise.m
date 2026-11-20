function g = FOTV_denoise(f,fidelity,mu,iter_num,gpu)

[sx,sy,sz]=size(f);
sizeg=[sx,sy,sz];

alpha = 1.6;
K = 15;

paral1 = 0;
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
%     bl1 =bx;
else
    bx = zeros(sizeg,'single');
    by = bx;
    bz = bx;
%     bl1 =bx;
end

g_update = single((fidelity/mu)*f);

for iter = 1:iter_num

    g_update = p2o(g_update);
    if iter>1
        g = real(o2p(g_update./eigA));
    else
        g = real(o2p(g_update./(fidelity/mu)));
    end
    
    g_update = (fidelity/mu).*f;
    
    [Lx,bx]=iter_xFotv(g,bx,wx,mu,alpha);
    g_update = g_update+Lx;
    
    [Ly,by]=iter_yFotv(g,by,wy,mu,alpha);
    g_update = g_update+Ly;
    
    [Lz,bz]=iter_zFotv(g,bz,wz,mu,alpha);
    g_update = g_update+Lz;
   
%     [Lsparse,bl1]=iter_sparse(g,bl1,mu,paral1);
%     g_update = g_update+Lsparse;

%     disp(['  iter ' num2str(iter) ' | ' num2str(iter_num) ', done! ']);
%     progressbar(iter/iteration);
end

g(g<0)=0;
clear bxx byy bzz bxz bxy byz bl1 f normlize g_update

end
