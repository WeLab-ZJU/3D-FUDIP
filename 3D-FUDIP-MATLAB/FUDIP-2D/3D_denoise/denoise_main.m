addpath('./BM4D/')
addpath('./BM4D/ACsN/')
addpath('./BM4D/NoiseEstimate/')
addpath('./Utils/')
addpath('./SparseDenoise/')
addpath('./SparseDenoise/Sparse_denoise/SHOperation');
addpath('./SparseDenoise/Sparse_denoise/SHIter');
addpath('./FOTV/')
addpath('./FOTV/Iter/')
%% load image
i = loadtiff('/mnt/dzh/MATLAB-decon/framelet_decon_v7/data/S1-2-5-crop1.tif');
data = single(i(230:230+225,500:500+225,21:100));
save_name = '/mnt/dzh/MATLAB-decon/framelet_decon_v7/results/fig_extend/S1-2-5-crop1_NLM_denoise.tif';
%% 
method = 'SparseDenoise';

disp('Start denoising...');
tic
switch method
    case 'BM4D'
        CoreNum=4; % set cpu core number
        if isempty(gcp('nocreate')) 
            parpool(CoreNum);
        end
        sigma = 0.05;
        out = BM4D_denoise(data,sigma);
    case 'NLM'
        sigma = 1;
        out = FastNonLocalMeans3D(data,sigma); 
    case 'FOTV'
        fidelity = 1000;
        mu = 20;
        iter_num = 100;
        gpu = 1;
        out = FOTV_denoise(data,fidelity,mu,iter_num,gpu);
        out = gather(out);
    case 'FOTVHessian'
        fidelity = 100;
        mu = 3;
        paral1 = 0;
        iter_num = 100;
        contiz = 0.5;
        gpu = 1;
        out = FOTV_Hessian_denoise(data,fidelity,mu,paral1,iter_num,contiz,gpu);
        out = gather(out);
    case 'SparseDenoise'
        fidelity = 120;
        sparsity = 0;
        zconti = 0.1;
        SHIter = 20;
        gpu = 1;
        out = sparse_denoise(data,fidelity,sparsity,zconti,SHIter,gpu);
        out = gather(out);
end
toc
% imshow3D(out,[])
figure(),
subplot(121), imshow(data(:, :, 30), [])
subplot(122), imshow(out(:, :, 30), [])
%saveastiff(single(out),save_name);