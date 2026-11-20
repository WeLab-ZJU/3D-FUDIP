%**************************************************
% Demo script for fudip2D
%**************************************************
clear
clc
close all

%% 0. Add paths
addpath('./framelet2D/')
addpath('./utils/')
addpath('./fudip2D/')
p = genpath('./3D_denoise/');
addpath(p);

%% 1. Data Preparation
file_path = '/data/demo_data/Synthetic_panel/';
save_path = '/results/';
file_name = 'synthetic_panel_degraded_input.tif';

%% 2. Parameters
% -------- Preprocessing --------
using_upsample = 0;
upscale_factor = 2;
bg_estimation  = 1;
bg_ratio       = 0.2;
pad_size       = 10;
optimize_padding = 1;
save_res       = 1;
psf_type       = 'generate';
denoise_method = 'Hessian';
% -------- ADMM parameters --------
useGPU         = 0;
use_accelerated = 1;  % 1: fudipCore2DProOR (iter=60); 0: fudipCore2DPro (iter=100)
if use_accelerated
    iter_num = 60;
    method_name = 'fudipCore2DProOR';
else
    iter_num = 100;
    method_name = 'fudipCore2DPro';
end
rho1            = 1;          % ADMM penalty parameter
rho2            = 0.1;          % ADMM penalty parameter
rho3            = 0.1;          % ADMM penalty parameter

alpha          = 3e-3;
beita          = 2e-4;

stop_criterion = 1e-6;
% -------- ProOR parameters --------
relaxation       = 1.7;    % Over-relaxation factor omega in (0, 2)
min_iter         = 2;      % Minimum iterations before early stopping
track_residuals  = false;  % Store primal residual history in diagnostics


switch psf_type
    case 'read'
        psf_path = 'path/to/psf';
        psf_name = 'psf.tif';
        psf = loadtiff(fullfile(psf_path, psf_name));
    case 'generate'
        Pixel = 108.3*10^-9;
        lambda = 510*10^-9;
        NA = 1.5;
        sampling_dx = Pixel / 2;
        psf = kernel(sampling_dx, lambda, NA);
end

%% 3. Load and preprocess image
raw_img = single(loadtiff(fullfile(file_path, file_name)));
raw_img = raw_img ./ max(raw_img(:));
f = max(raw_img, 0);


psf = psf ./ sum(psf(:));
f = edgetaper(f, psf);
f = f ./ max(f(:));

if bg_estimation == 1
    bg = single(background_estimation(f) .* bg_ratio);
else
    bg = f .* 0;
end

if using_upsample == 1
    f = Fourier_Oversample(f, upscale_factor);
    psf = Fourier_Oversample(psf, upscale_factor);
    bg = imresize(bg, upscale_factor);
end

effective_pad_size = getOptimizedPadSize2D(size(f), pad_size, optimize_padding);
if effective_pad_size > 0
    % f = padarray(f, [effective_pad_size effective_pad_size], 'symmetric', 'both');
    % bg = padarray(bg, [effective_pad_size effective_pad_size], 'symmetric', 'both');
    f = padarray(f, [effective_pad_size effective_pad_size], 0, 'both');
    bg = padarray(bg, [effective_pad_size effective_pad_size], 0, 'both');
end

[img_h, img_w] = size(f);
psf = alignSize2D(psf, img_h, img_w);
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
        args.iter_num = 2;
        args.gpu = useGPU;
        args.zconti = 0.05;
        denoiser = @SparseHessian_wraper_gpu;
    case 'BM4D'
        args.sigma = 0.01;
        denoiser = @BM4D_wraper;
    otherwise
        error('Unknown denoiser: %s', denoise_method);
end

%% 5. Pack options and run over-relaxed ADMM deconvolution
options.useGPU           = useGPU;
options.iter_num         = iter_num;
options.rho1              = rho1;
options.rho2              = rho2;
options.rho3              = rho3;

options.alpha            = alpha;
options.beita            = beita;
options.stop_criterion   = stop_criterion;
options.frame            = 2;       % Piecewise Linear Framelet
options.Level            = 3;       % 3-level decomposition
options.verbose          = true;
options.relaxation       = relaxation;
options.min_iter         = min_iter;
options.track_residuals  = track_residuals;

if useGPU
    gpu = gpuDevice(1);
    reset(gpu); wait(gpu);
    if options.verbose
        disp(['GPU free memory: ', num2str(gpu.FreeMemory/1024/1024), ' MB']);
    end
end

if use_accelerated
    [x, RelErr, diag_or] = fudipCore2DProOR(f, psf, bg, denoiser, args, options);
else
    [x, RelErr] = fudipCore2DPro(f, psf, bg, denoiser, args, options);
end

x_crop = removePadding2D(x, effective_pad_size);
f_crop = removePadding2D(f, effective_pad_size);

%% 6. Visualization 
%{
figure
subplot(1,2,1), imshow(f_crop, []), title('Input')
if use_accelerated
    subplot(1,2,2), imshow(x_crop, []), title('Deconvolved (ProOR)')
    fprintf('[%s] Iters = %d | Stopped = %d | Final NSDE = %.2e\n', ...
        method_name, length(RelErr), diag_or.stopped, RelErr(end));
else
    subplot(1,2,2), imshow(x_crop, []), title('Deconvolved')
    fprintf('[%s] Iters = %d | Final NSDE = %.2e\n', ...
        method_name, length(RelErr), RelErr(end));
end

figure
plot(RelErr(2:end))
xlabel('Iteration')
ylabel('NSDE')
if use_accelerated
    title('Convergence (ProOR)')
else
    title('Convergence')
end
%}

%% 7. Save results and log
[~, file_base_name, file_ext] = fileparts(file_name);
if using_upsample == 1
    up_str = sprintf('_up_%d', upscale_factor);
else
    up_str = '';
end

if isfield(args, 'iter_num')
    hessian_iter_str = sprintf('_Hiter_%d', args.iter_num);
else
    hessian_iter_str = '';
end

save_name = 'fudip_recon_output.tif';
log_name= 'fudip_recon_logs.log';
%save_name = sprintf('%s_iter_%d_rho1_%g_rho2_%g_rho3_%g_alpha_%g_beita_%g%s%s', ...
%    file_base_name, iter_num, rho1, rho2, rho3,alpha, beita, hessian_iter_str, file_ext);
%log_name = sprintf('%s_iter_%d_rho1_%g_rho2_%g_rho3_%g_alpha_%g_beita_%g%s.log', ...
%    file_base_name, iter_num, rho1, rho2, rho3,alpha, beita, hessian_iter_str);

if ~exist(save_path, 'dir')
    try
        mkdir(save_path);
    catch
        disp(['Warning: Cannot create directory ', save_path]);
    end
end

log_fid = fopen(fullfile(save_path, log_name), 'w');
if log_fid ~= -1
    fprintf(log_fid, '--- FUDIP 2D Deconvolution Parameter Settings ---\n');
    fprintf(log_fid, 'File path: %s\n', file_path);
    fprintf(log_fid, 'File name: %s\n', file_name);
    fprintf(log_fid, 'Save path: %s\n', save_path);
    fprintf(log_fid, 'Save name: %s\n\n', save_name);

    fprintf(log_fid, '--- Main Parameters ---\n');
    fprintf(log_fid, 'use_accelerated: %d\n', use_accelerated);
    fprintf(log_fid, 'method: %s\n', method_name);
    fprintf(log_fid, 'using_upsample: %d\n', using_upsample);
    if using_upsample == 1
        fprintf(log_fid, 'upscale_factor: %d\n', upscale_factor);
    end
    fprintf(log_fid, 'deconv_input_size: [%d %d]\n', img_h, img_w);
    fprintf(log_fid, 'bg_estimation: %d\n', bg_estimation);
    fprintf(log_fid, 'bg_ratio: %g\n', bg_ratio);
    fprintf(log_fid, 'pad_size: %d\n', pad_size);
    fprintf(log_fid, 'optimize_padding: %d\n', optimize_padding);
    fprintf(log_fid, 'effective_pad_size: %d\n', effective_pad_size);
    fprintf(log_fid, 'denoise_method: %s\n', denoise_method);
    fprintf(log_fid, 'iter_num (max): %d\n', iter_num);
    fprintf(log_fid, 'rho1: %g\n', rho1);
    fprintf(log_fid, 'rho2: %g\n', rho2);
    fprintf(log_fid, 'rho3: %g\n', rho3);
    fprintf(log_fid, 'alpha: %g\n', alpha);
    fprintf(log_fid, 'beita: %g\n', beita);
    fprintf(log_fid, 'stop_criterion: %g\n', stop_criterion);
    fprintf(log_fid, 'frame: %d\n', options.frame);
    fprintf(log_fid, 'Level: %d\n\n', options.Level);

    if use_accelerated
        fprintf(log_fid, '--- ProOR Parameters ---\n');
        fprintf(log_fid, 'relaxation: %g\n', relaxation);
        fprintf(log_fid, 'min_iter: %d\n', min_iter);
        fprintf(log_fid, 'track_residuals: %d\n\n', track_residuals);
    end

    fprintf(log_fid, '--- Denoiser Args (%s) ---\n', denoise_method);
    if exist('args', 'var')
        args_fields = fieldnames(args);
        for i_field = 1:length(args_fields)
            fprintf(log_fid, '%s: %g\n', args_fields{i_field}, args.(args_fields{i_field}));
        end
    end

    fprintf(log_fid, '\n--- Run Summary ---\n');
    fprintf(log_fid, 'actual_iters: %d\n', length(RelErr));
    if use_accelerated
        fprintf(log_fid, 'stopped: %d\n', diag_or.stopped);
    end
    fprintf(log_fid, 'final_NSDE: %.6e\n', RelErr(end));
    fprintf(log_fid, 'Deconvolution done!\n');
    fclose(log_fid);
end

if save_res == 1
    saveastiff(single(x_crop), fullfile(save_path, save_name));
    saveastiff(single(raw_img), fullfile(save_path, 'raw_input.tif'))
end


function effective_pad_size = getOptimizedPadSize2D(img_size, pad_size, optimize_padding)
%GETOPTIMIZEDPADSIZE2D Keep FFT-friendly image sizes unchanged when padding is optional.

if optimize_padding == 1 && isPowerOfTwo(img_size(1:2))
    effective_pad_size = 0;
else
    effective_pad_size = pad_size;
end
end

function tf = isPowerOfTwo(value)
%ISPOWEROFTWO Return true for positive power-of-two integer sizes.

tf = all(value > 0 & abs(value - round(value)) < 1e-6 & 2.^nextpow2(value) == value);
end

function img = removePadding2D(img, pad_size)
%REMOVEPADDING2D Remove symmetric 2D padding when padding was applied.

if pad_size > 0
    img = img(pad_size+1:end-pad_size, pad_size+1:end-pad_size);
end
end
