clear all
clc
%% Load image 
%{
% test image: PSF-CElegans-CY3 run successfully
psfname = 'D:\Download\Compressed\PSF-CElegans-CY3\PSF-CElegans-CY3.tif';
dataname = 'D:\Download\Compressed\CElegans-CY3\CElegans-CY3.tif';
addpath('D:\code\MATLAB\GlobalBioIm-master\Util\')

psf=double(loadtiff(psfname));psf=psf/sum(psf(:));
y=double(loadtiff(dataname));maxy=max(y(:));y=y/maxy;

%}

% sz=size(y);
% sp =size(psf);
% 
% padsz=[0,0,52];
% sznew=fft_best_dim(sz+2*padsz);
% halfPad=(sznew-sp)/2;
% psf=padarray(psf,halfPad,0,'both');
% img = zeros(720,720,216);

% i = loadtiff('E:/dataset/kindney/raw/low_2.tif');
% imgs = i(100:500,100:500,101:251);%(1000:1500,1000:1500,1:50)
% img=padarray(imgs,[10,10,10],0,'both');
% 
% psfs = loadtiff('C:\Users\Administrator\Desktop\psf\SPIM_psf1.tif');
% sznew = size(img);
% sp = size(psfs);
% halfPad=(sznew-sp)/2;
% psf=padarray(psfs,halfPad,0,'both');

i = loadtiff('E:\dataset\kindney\raw\low_5.tif');
img = single(i(100:601,100:601,100:301));%(1000:1500,1000:1500,1:50)

psfs = loadtiff('C:\Users\Administrator\Desktop\psf\psf_nqst.tif');
psf = psfs(254:end-253,254:end-253,60:end-59);
psf = psf./max(psf(:));
%
% i = loadtiff('E:/dataset/kindney/raw/low_2.tif');
% psf = loadtiff('D:/code/MATLAB/sparse_Framelet/psf/new_psf/psf.tif');
% i = loadtiff('D:\code\Julia\ThreeDeconv.jl\example\datasetwithraw\raw\bead\ffc_cropped\bead_lowsnr_ffcor.tif');
% psfs = loadtiff('D:/code/Julia/ThreeDeconv.jl/example/psf.tif');
% psf = psfs(5:end-6,5:end-6,5:end-6);
%

% img = i;%(1000:1500,1000:1500,1:50)
fidelity = 100;
contiz = 0.5;
paral1 = 0;
iteration = 2;
gpu = 1;
mu = 0.5;

g = sparse_hessian_deconv(img,psf,fidelity,contiz,paral1,iteration,gpu,mu);

g = max(0,g);
figure,
subplot(121),imshow(real(img(:,:,100)),[])
subplot(122),imshow((g(:,:,100)),[])