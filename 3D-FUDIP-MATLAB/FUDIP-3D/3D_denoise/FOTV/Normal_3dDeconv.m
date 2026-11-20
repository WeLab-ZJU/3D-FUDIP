close all
clear 
clc
%% load image
% widefield image 
i = loadtiff('E:\data\hlm\denonv_test2\denonv_test2_crop.tif');
% data = single(i(1:1601,1:1601,1:end));%(1000:1500,1000:1500,1:50)
data = single(i(151:650,151:650,:));%(1000:1500,1000:1500,1:50)
% backgrounds = background_estimation(data./2);
backgrounds = 0 ;
img = data - backgrounds;
img = img./max(img(:));

psfs = loadtiff('D:\code\MATLAB\Big_data_deconv\psf\spinning_disk_psf.tif');
psf = psfs(1:end-1,1:end-1,:);
% psf = padarray(psfs,[400 400],0,'both');
psf=psf./sum(psf(:));

%% set parameters
useGPU = 1;
[M,N,P]=size(img);
imageSize = size(img); 
psf = Align_size(psf,M,N,P);

mu1 = 120;
alpha = 30; % balance x and g
stop_criterion = 0.5;

iternum = 30;
f = single(img);
psf = single(psf);
z = single(zeros([imageSize,3]));

if useGPU
    f = gpuArray(f);    
    psf = gpuArray(psf);
    z = gpuArray(z);
end
OTF = psf2otf(psf,imageSize);

cFT = OTF;
cTFT = conj(OTF);
eig_A = mu1.*(cTFT.*cFT) + alpha;
clear psf cTFT
% initialize intermediate variables
g = f;

r = zeros(iternum); % iteration information

p2o = @(x) psf2otf(x, imageSize);

x_old = gather(f);

for iter = 1:iternum
    %% deblur step
    if iter > 1
        x_old = gather(x);
    end
    % update x
    x = otf2psf((alpha*p2o(g) + mu1.*conj(OTF).*p2o(f))./eig_A);  
    x = real(x);
    % update g 
    g = FOTV_denoise(x);
    
    delta = sqrt(sum((gather(x(:)) - x_old(:)).^2));
    if iter == 1
        delta_rel = 0;
    else
        delta_rel = abs((deltaL - delta) / deltaL * 100);    
    end
    x_old = gather(x);
    deltaL = delta;
    disp(['iteration: ' num2str(iter), ' delta: ' num2str(delta_rel, 3)]);  
    if delta_rel < 0.1 && iter > 1
        break;
    end
end

x(x<0) = 0;
out = gather(x);
subplot(121),imshow(img(:,:,28),[])
subplot(122),imshow(out(:,:,28),[])

% saveastiff(img,'E:\data\hlm\dzh20211026\CA1-1-WF-40X_01_crop1.tif')
% saveastiff(out,'E:\data\hlm\dzh20211026\CA1-1-WF-40X_01_crop1deconv.tif')
% saveastiff(single(out),'E:\data\hlm\denonv_test2\denonv_test2_normal_deconvx3.tif');