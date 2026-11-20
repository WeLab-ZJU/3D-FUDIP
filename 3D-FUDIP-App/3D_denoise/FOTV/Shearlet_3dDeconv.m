clear 
clc
% warning('off')
%% load image
% widefield image 
i = loadtiff('E:\data\hlm\denonv_test2\denonv_test2_crop.tif');
data = single(i(151:650,151:650,:));%(1000:1500,1000:1500,1:50)
% backgrounds = background_estimation(data./2);
backgrounds = 0 ;
img = data - backgrounds;
img = img./max(img(:));

psfs = loadtiff('D:\code\MATLAB\Big_data_deconv\psf\spinning_disk_psf.tif');
psf = psfs(1:end-1,1:end-1,:);
psf=psf./sum(psf(:));
%% set parameters
useGPU = 1;
scales = 1;
Length = 50;
[M,N,P]=size(img);
imageSize = size(img); 
p2o = @(x) psf2otf(x, imageSize);
psf = Align_size(psf,M,N,P);
tic;
mu1 = 120;
alpha1 = 120; % balance x and g
alpha3 = 2;
lamda1 = 1;

stop_criterion = 0.1;

iternum = 30;
f = single(img);
psf = single(psf);

w1 = single(zeros(M,N,P,Length));

if useGPU
    f = gpuArray(f);    
    psf = gpuArray(psf);
%     w1 = gpuArray(w1);
end

b1 = w1;

OTF = single(p2o(psf));
cOTF = conj(OTF);
eig_A = mu1.*(OTF.*cOTF) + alpha1;
clear psf cOTF
% initialize intermediate variables
g = f;
r = zeros(iternum); % iteration information

x_old = gather(f);

for iter = 1:iternum
    %% deblur step
    if iter > 1
        x_old = gather(x);
    end
    % update x
    [eig_C,eig_S] = get_shearlet(w1,b1,scales);
    x = otf2psf((alpha1*p2o(g) + mu1.*conj(OTF).*p2o(f) + alpha3*conj(p2o(eig_C)))./(eig_A+eig_S));  
    x = real(x);
    clear eig_C eig_S
    
    [w1,b1] =  update_shearlet(x,w1,b1,lamda1,alpha3,scales);
%     w1 = shrink_operator(eig_C,b1,lamda1./alpha3);
%     b1 = b1 + eig_C - w1 ;
%     clear eig_C 
    % update g 
%     g = Sparse_denoise(x);
    g = FOTV_denoise(x);
    
    delta = sqrt(sum((gather(x(:)) - x_old(:)).^2));
    if iter == 1
        delta_rel = 0;
    else
        delta_rel = abs((deltaL - delta) / deltaL * 100);    
    end
    x_old = gather(x);
    figure(),imshow(gather(x_old),[])
    deltaL = delta;
    disp(['iteration: ' num2str(iter), ' delta: ' num2str(delta_rel, 3)]); 
    if delta_rel < 0.08 && iter > 1
        break;
    end
end
x(x<0)=0;
out = gather(x);
subplot(121),imshow(img(:,:,13),[])
subplot(122),imshow(out(:,:,13),[])
toc;
%saveastiff(single(out),'E:\data\hlm\denonv_test2\denonv_test2_crop_shearlet_deconv3.tif');