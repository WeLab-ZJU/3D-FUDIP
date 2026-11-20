function [x, RelErr] = fudipCore3DPro(f, psf, bg, denoiser, denoiser_args, options)
% FUDIPCORE3D  ADMM-based Poisson decoupled deconvolution for 3D
%   fluorescence microscopy image volumes.
%
% USAGE:
%   [x, RelErr] = fudipCore3D(f, psf, bg, denoiser, denoiser_args)
%   [x, RelErr] = fudipCore3D(f, psf, bg, denoiser, denoiser_args, options)
%
% DESCRIPTION:
%   This function implements the ADMM (Alternating Direction Method of
%   Multipliers) iterative framework for Poisson-noise deconvolution with
%   framelet regularization and a plug-and-play denoiser prior.
%
% INPUT:
%   f              - 3D observed image volume (single precision recommended)
%   psf            - 3D point spread function, must match the size of f
%   bg             - 3D background estimate, same size as f
%   denoiser       - function handle for the denoiser, called as
%                    denoiser(image, denoiser_args)
%   denoiser_args  - struct of parameters passed to the denoiser
%   options        - (optional) struct with algorithm parameters:
%       .iter_num       - maximum number of ADMM iterations (default: 300)
%       .rho            - fallback ADMM penalty for rho1/rho2/rho3
%                         (default: 1)
%       .rho1           - ADMM penalty for Poisson data-fidelity constraint
%                         z1 = Hx + b (default: options.rho)
%       .rho2           - ADMM penalty for denoiser coupling constraint
%                         z2 = x (default: options.rho)
%       .rho3           - ADMM penalty for framelet sparsity constraint
%                         z3 = Wx (default: options.rho)
%       .alpha          - trade-off weight between deblurred and denoised
%                         images (default: 2e-4)
%       .beita          - framelet sparsity regularization weight
%                         (default: 1e-4)
%       .stop_criterion - relative error threshold for early stopping
%                         (default: 2e-5)
%       .frame          - framelet type following the 2D convention
%                         (1: Haar, 2: piecewise linear, 3: piecewise cubic)
%                         (default: 2)
%       .Level          - framelet decomposition levels (default: 4)
%       .useGPU         - whether to use GPU acceleration (default: 1)
%       .verbose        - whether to print iteration info (default: true)
%
% OUTPUT:
%   x       - deconvolved image volume, non-negative and normalized to [0, 1]
%   RelErr  - vector of relative error (NSDE) at each iteration

%% Parse options with defaults
if nargin < 6, options = struct(); end

iter_num       = getOpt(options, 'iter_num',       300);
rho            = getOpt(options, 'rho',            1);
rho1           = getOpt(options, 'rho1',           rho);
rho2           = getOpt(options, 'rho2',           rho);
rho3           = getOpt(options, 'rho3',           rho);
alpha          = getOpt(options, 'alpha',          2e-4);
beita          = getOpt(options, 'beita',          1e-4);
stop_criterion = getOpt(options, 'stop_criterion', 2e-5);
frame          = getOpt(options, 'frame',          2);
Level          = getOpt(options, 'Level',          4);
useGPU         = getOpt(options, 'useGPU',         1);
verbose        = getOpt(options, 'verbose',        true);

%% Ensure single precision
f   = single(f);
psf = single(psf);
bg  = single(bg);

%% Initialize framelet operators
[D, R] = GenerateFrameletFilter3D(frame);
W  = @(x) FraDecMultiLevel3D(x, D, Level);
WT = @(x) FraRecMultiLevel3D(x, R, Level);

%% Allocate ADMM auxiliary variables
fsize = size(f);

z   = single(zeros([fsize, 2]));
v   = single(zeros([fsize, 2]));
v_3 = W(zeros(fsize, 'single'));

%% GPU initialization
if useGPU
    f   = gpuArray(f);
    psf = gpuArray(psf);
    z   = gpuArray(z);
    v   = gpuArray(v);
    bg  = gpuArray(bg);
    v_3 = W(gpuArray(zeros(fsize, 'single')));
end

%% Precompute OTF and frequency-domain denominator
OTF  = single(p2o(psf));
cOTF = conj(OTF);
AtA  = rho1.*(OTF .* cOTF) + rho2 + rho3;

%% Initialize primal variables
x     = f;
g     = f;
x_old = f;

RelErr = double(zeros(iter_num, 1));

%% ADMM iterations
tic
for iter = 1:iter_num
    if iter > 1
        x_old = x;
    end

    % --- Update z1 (Poisson data-fidelity proximal) ---
    Ax = real(ifftn(OTF .* fftn(x)));
    t1 = Ax + bg + v(:,:,:,1) - 1./rho1;
    t2 = sqrt(t1.^2 + 4.*f./rho1);
    z(:,:,:,1) = 0.5 .* (t1 + t2);

    % --- Update z2 (denoiser coupling proximal) ---
    z(:,:,:,2) = (alpha.*g + rho2.*(x + v(:,:,:,2))) ./ (alpha + rho2);

    % --- Update z3 (framelet sparsity proximal) ---
    z_3 = ShrinkFramelet3D(AddFrameletArray3D(W(x), v_3), beita./rho3);

    % --- Update x (linear system solve in Fourier domain) ---
    WtW = WT(SubFrameletArray3D(z_3, v_3));
    XtX = rho1.*cOTF.*fftn(z(:,:,:,1) - v(:,:,:,1) - bg) ...
        + rho2.*fftn(z(:,:,:,2) - v(:,:,:,2)) ...
        + rho3.*fftn(WtW);
    x = single(real(ifftn(XtX ./ AtA)));

    % --- Update dual variables v1, v2 ---
    v(:,:,:,1) = v(:,:,:,1) + real(ifftn(OTF .* fftn(x))) + bg - z(:,:,:,1);
    v(:,:,:,2) = v(:,:,:,2) + x - z(:,:,:,2);

    % --- Update dual variable v3 ---
    v_3 = SubFrameletArray3D(AddFrameletArray3D(W(x), v_3), z_3);

    % --- Denoise step (plug-and-play prior) ---
    if useGPU
        g = gpuArray(single(denoiser(gather(x), denoiser_args)));
    else
        g = single(denoiser(x, denoiser_args));
    end

    % --- Convergence check ---
    if iter == 1
        NSDE = 0;
    else
        NSDE = norm(x(:) - x_old(:), 'fro') / norm(x(:), 'fro');
    end

    if useGPU
        NSDE = gather(NSDE);
    end
    RelErr(iter) = double(NSDE);

    if verbose
        disp(['iteration: ', num2str(iter), '  NSDE: ', num2str(NSDE)]);
    end

    if iter > 1 && abs(NSDE) <= stop_criterion
        if verbose
            disp('Stop criterion reached. Finishing iterations.');
        end
        break
    end
end
elapsed = toc;

if verbose
    disp(['Total elapsed time: ', num2str(elapsed, '%.2f'), ' s']);
end

%% Trim unused RelErr entries and gather result
RelErr = RelErr(1:iter);
if useGPU
    x = gather(max(0, x));
else
    x = max(0, x);
end
x = x ./ max(x(:));

end

%% ======================== Local helper ========================
function val = getOpt(s, field, default)
% Retrieve a field from struct s, or return the default value.
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
