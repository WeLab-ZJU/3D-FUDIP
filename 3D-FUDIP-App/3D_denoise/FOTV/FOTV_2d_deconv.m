%% data preparation
f = imread('./data/lenna.bmp');
f = single(f);
I = f./max(f(:));

f_size = size(f);
f = imnoise(I,'gaussian',0,0.01);
raw = f;
%{
% Peak
peak = 25.5;
I = I/max(I(:))*peak;

% Adding blur
H = fspecial('gaussian',9,sqrt(3)); % blurring kernel
I_blurry = imfilter(I,H,'circular');


% Adding Poisson noise
f = poissrnd(I_blurry);
raw = f;
%}
%% fraction order setting
alpha = 1.6;
K = 13; 
beta = [1.3,10.3,0.5];
paral1 = 0;
fidelity = 20;
iter_num =50;
mu = 1;

gpu = 1;

w = zeros(1,K);w(K) = 1;

for i = 1:(K-1)
    w(1,K-i) = (-1)^i*gamma(alpha+1)/gamma(alpha-i+1)/factorial(i);
end

xtx = defDtD(w,f_size);
yty = defDtD(w',f_size);


eigDtD = beta(1).*xtx + beta(2).*yty;
% eigDtD = mu.*(xtx+yty);
eigA = single((fidelity/mu) + paral1^2 + eigDtD);

clear xtx yty eigDtD 

if gpu==1
    f=gpuArray(f);
    eigA =gpuArray(eigA);
    bx = gpuArray.zeros(f_size,'single');
    by = bx;
    bl1 =bx;
else
    bx = zeros(sizeg,'single');
    by = bx;
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
    
    [Lx,bx]=iter_xFotv(g,bx,w,mu,alpha);
    g_update = g_update+Lx;
    
%     figure,imshow(gather(Lx)),title('Lx')
    [Ly,by]=iter_yFotv(g,by,w',mu,alpha);
    g_update = g_update+Ly;
%     figure,imshow(gather(Ly)),title('Ly')
    
    [Lsparse,bl1]=iter_sparse(g,bl1,mu,paral1);
    g_update = g_update+Lsparse;

    disp(['  iter ' num2str(iter) ' | ' num2str(iter_num) ', done! ']);
%     progressbar(iter/iteration);
end

g(g<0)=0;

clear bxx byy bzz bxz bxy byz bl1 f normlize g_update

res = gather(g);
figure()
subplot(121),imshow(raw,[])
subplot(122),imshow(res,[])
