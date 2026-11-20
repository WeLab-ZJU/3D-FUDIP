%**************************************************
% Demo script for fudip3D
%**************************************************
clear
clc
close all

%% 0. Add paths
addpath('./framelet3D/')
addpath('./utils/')
addpath('./fudip3D/')
p = genpath('./3D_denoise/');
addpath(p);

%% 1. Data Preparation
file_path = 'D:/Results/ExampleData/liveCell-neuron/data1/';
save_path = 'D:/Results/ExampleData/example_output/';
file_name = 'liveCell_input1.tif';
psf_path = 'D:/Results/ExampleData/liveCell-neuron/data1/';
psf_name = 'psf_liveCell1_crop.tif';
name_prefix = '';

%% 2. Parameters
pad_xy = 20;
pad_z = 10;

psf_xy_crop = 30;
psf_z_crop = 0;

bg_estimation = 1;
bg_ratio = 0.6;

save_res = 1;
denoise_method = 'Hessian';

% upsample factor
upscale_factor_xy = 1;
upscale_factor_z = 1;

psf_upscale_ratio_xy = 1.0;
psf_upscale_ratio_z = 1.0;

% framelet
framelet_frame = 2;
framelet_level = 2;

% ADMM iteration parameters
useGPU = 1;
use_accelerated = 1;  % 1: fudipCore3DProOR (iter=60); 0: fudipCore3DPro (iter=100)
if use_accelerated
    iter_num = 60;
    method_name = 'fudipCore3DProOR';
else
    iter_num = 100;
    method_name = 'fudipCore3DPro';
end
rho1 = 1;
rho2 = 0.1;
rho3 = 0.1;
alpha = 1e-5;
beita = 2e-4;
stop_criterion = 0.0001;

%% 3. Load and preprocess image
disp('============================================');
disp('Step 1: Loading raw data...');
i = loadtiff([file_path, file_name]);
img = single(i);
img = img ./ max(img(:));
f = max(single(img), 0);

disp('Step 2: Substracting background...');
if bg_estimation == 1
    bg = single(background_estimation(img).* bg_ratio) ;
else
    bg = img .* 0;
end

disp('Step 3: Preparing psf...');
psf = single(loadtiff([psf_path, psf_name]));
psf = psf(psf_xy_crop+1:end-psf_xy_crop,psf_xy_crop+1:end-psf_xy_crop,psf_z_crop+1:end-psf_z_crop);

if (upscale_factor_xy - 1) ~= 0 || (upscale_factor_z - 1) ~= 0
    [size_y, size_x, size_z] = size(f); 
    output_size = ceil([size_y *upscale_factor_xy, size_x * upscale_factor_xy, size_z *upscale_factor_z]); 
    f = single(imresize3(f, output_size));
    % f = Fourier_Oversample(f, upscale_factor_xy);
    
    [size_y, size_x, size_z] = size(psf);
    output_size = ceil([size_y * upscale_factor_xy, size_x * upscale_factor_xy, size_z * upscale_factor_z]);
    psf = single(imresize3(psf, output_size));
    
    [size_y, size_x, size_z] = size(bg);
    output_size = ceil([size_y * upscale_factor_xy, size_x * upscale_factor_xy, size_z * upscale_factor_z]);
    bg = single(imresize3(bg, output_size));
end

f = padarray(f, [pad_xy pad_xy pad_z], 'symmetric', 'both');
bg = padarray(bg, [pad_xy pad_xy pad_z], 'symmetric', 'both');

% Auto-shrink for edgetaper: size(psf_taper) < size(f)/2 along each dim
max_taper = floor((size(f) - 1) / 2);
taper_m = max(0, ceil((size(psf) - max_taper) / 2));
taper_m = min(taper_m, floor((size(psf) - 1) / 2));
psf_taper = psf(1+taper_m(1):end-taper_m(1), ...
    1+taper_m(2):end-taper_m(2), ...
    1+taper_m(3):end-taper_m(3));
psf_taper = psf_taper ./ sum(psf_taper(:));
f = edgetaper(f, psf_taper);
f = f ./ max(f(:));

[img_h, img_w, img_d] = size(f);

psf = alignSize3D(psf, img_h, img_w, img_d);
psf = max(psf, 0);
psf = psf ./ sum(psf(:));
%% 4. Choose denoiser
switch denoise_method
    case 'FOTV'
        args.fidelity = 1000;
        args.mu = 25;
        args.iter_num = 5;
        args.gpu = useGPU;
        denoiser = @FOTV_wraper;
    case 'FOTVHessian'
        args.fidelity = 1000;
        args.mu = 20;
        args.iter_num = 1;
        args.gpu = useGPU;
        args.contiz = 0.3;
        denoiser = @FOTVHessian_wraper;
    case 'Hessian'
        args.fidelity = 100;
        args.sparsity = 8;
        args.iter_num = 1;
        args.gpu = useGPU;
        args.zconti = 0.05;
        denoiser = @SparseHessian_wraper_gpu;
    case 'BM4D'
        args.sigma = 0.01;
        denoiser = @BM4D_wraper;
    otherwise
        error('Unknown denoiser: %s', denoise_method);
end

%% 5. Pack options and run ADMM deconvolution
options.useGPU         = useGPU;
options.iter_num       = iter_num;
options.rho1           = rho1;
options.rho2           = rho2;
options.rho3           = rho3;
options.alpha          = alpha;
options.beita          = beita;
options.stop_criterion = stop_criterion;
options.frame          = framelet_frame;       % Piecewise Linear Framelet
options.Level          = framelet_level;       % 2-level decomposition
options.verbose        = true;

if useGPU
    gpu = gpuDevice(1);
    reset(gpu); wait(gpu);
    if options.verbose
        disp(['GPU free memory: ', num2str(gpu.FreeMemory/1024/1024), ' MB']);
    end
end
disp('Step 4: Starting deconvolution...');
if use_accelerated
    [x, RelErr, ~] = fudipCore3DProOR(f, psf, bg, denoiser, args, options);
else
    [x, RelErr] = fudipCore3DPro(f, psf, bg, denoiser, args, options);
end

%% 6. Visualization
% Check if the image retains its original scale in all spatial dimensions
if upscale_factor_xy == 1 && upscale_factor_z == 1
    % Apply standard cropping using the original padding size
    f = f(pad_xy+1:end-pad_xy, pad_xy+1:end-pad_xy, pad_z+1:end-pad_z);
    x = x(pad_xy+1:end-pad_xy, pad_xy+1:end-pad_xy, pad_z+1:end-pad_z);
else
    % Calculate dynamic padding sizes for axial and lateral dimensions
    % based on their respective upscaling factors
    % pad_xy = pad_xy * upscale_factor_xy;
    % pad_z  = pad_z * upscale_factor_z;
    
    % Validate if the calculated padding sizes are integers to ensure correct cropping
    if mod(pad_xy, 1) ~= 0 || mod(pad_z, 1) ~= 0
        disp('Warning: Pad size may not be cropped correctly due to non-integer scaling factors.')
    end
    
    % Crop the arrays dynamically across x, y, and z dimensions
    f = f(pad_xy+1:end-pad_xy, pad_xy+1:end-pad_xy, pad_z+1:end-pad_z);
    x = x(pad_xy+1:end-pad_xy, pad_xy+1:end-pad_xy, pad_z+1:end-pad_z);
end

figure
subplot(1,2,1), imshow(max(f, [], 3), []), title('Input (MIP)')
subplot(1,2,2), imshow(max(x, [], 3), []), title('Deconvolved (MIP)')
disp('Deconvolution done!')

% figure
% plot(RelErr1(2:end), 'r-')
% hold on 
% plot(RelErr2(2:end), 'b')
% xlabel('Iteration')
% ylabel('NSDE')
% title('Convergence')

%% 7. Save results and log
[~, file_base_name, file_ext] = fileparts(file_name);
up_str = '';

if isfield(args, 'iter_num')
    hessian_iter_str = sprintf('_Hiter_%d', args.iter_num);
else
    hessian_iter_str = '';
end

save_name = sprintf('%s_%s_iter_%d_upsample_%d_rho1_%g_rho2_%g_rho3_%g_alpha_%g_beita_%g%s%s', ...
    file_base_name, name_prefix, iter_num, upscale_factor_xy, rho1, rho2, rho3,alpha, beita, hessian_iter_str, file_ext);
log_name = sprintf('%s_iter_%d_rho1_%g_rho2_%g_rho3_%g_alpha_%g_beita_%g%s.log', ...
    file_base_name, iter_num, rho1, rho2, rho3,alpha, beita, hessian_iter_str);

if ~exist(save_path, 'dir')
    try
        mkdir(save_path);
    catch
        disp(['Warning: Cannot create directory ', save_path]);
    end
end

log_fid = fopen(fullfile(save_path, log_name), 'w');
if log_fid ~= -1
    fprintf(log_fid, '--- FUDIP 3D Deconvolution Parameter Settings ---\n');
    fprintf(log_fid, 'File path: %s\n', file_path);
    fprintf(log_fid, 'File name: %s\n', file_name);
    fprintf(log_fid, 'PSF path: %s\n', psf_path);
    fprintf(log_fid, 'PSF name: %s\n', psf_name);
    fprintf(log_fid, 'Save path: %s\n', save_path);
    fprintf(log_fid, 'Save name: %s\n\n', save_name);
    fprintf(log_fid, 'upscale_factor_xy: %g\n', upscale_factor_xy);
    fprintf(log_fid, 'upscale_factor_z: %g\n\n', upscale_factor_z);

    fprintf(log_fid, '--- Main Parameters ---\n');
    fprintf(log_fid, 'use_accelerated: %d\n', use_accelerated);
    fprintf(log_fid, 'method: %s\n', method_name);
    fprintf(log_fid, 'pad_size: %d\n', pad_xy);
    fprintf(log_fid, 'bg_estimation: %d\n', bg_estimation);
    fprintf(log_fid, 'denoise_method: %s\n', denoise_method);
    fprintf(log_fid, 'iter_num (main): %d\n', iter_num);
    fprintf(log_fid, 'rho1: %g\n', rho1);
    fprintf(log_fid, 'rho2: %g\n', rho2);
    fprintf(log_fid, 'rho3: %g\n', rho3);
    fprintf(log_fid, 'alpha: %g\n', alpha);
    fprintf(log_fid, 'beita: %g\n', beita);
    fprintf(log_fid, 'frame: %d\n', options.frame);
    fprintf(log_fid, 'Level: %d\n\n', options.Level);

    fprintf(log_fid, '--- Denoiser Args (%s) ---\n', denoise_method);
    if exist('args', 'var')
        args_fields = fieldnames(args);
        for i_field = 1:length(args_fields)
            fprintf(log_fid, '%s: %g\n', args_fields{i_field}, args.(args_fields{i_field}));
        end
    end
    fclose(log_fid);
end


if save_res == 1
    saveastiff(single(x), [save_path, save_name]);
end

