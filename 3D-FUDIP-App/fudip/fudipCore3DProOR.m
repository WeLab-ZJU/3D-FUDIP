function [x, RelErr, diagnostics] = fudipCore3DProOR(f, psf, bg, denoiser, denoiser_args, options)
% FUDIPCORE3DPROOR  Over-relaxed ADMM Poisson deconvolution for 3D volumes.
%
% USAGE:
%   [x, RelErr] = fudipCore3DProOR(f, psf, bg, denoiser, denoiser_args)
%   [x, RelErr] = fudipCore3DProOR(f, psf, bg, denoiser, denoiser_args, options)
%   [x, RelErr, diagnostics] = fudipCore3DProOR(...)
%
% DESCRIPTION:
%   This function mirrors fudipCore3DPro.m and adds over-relaxation to the
%   z-first ADMM update order. The z-subproblems are unchanged. Relaxed
%   auxiliary variables are used in the Fourier-domain x-update and in the
%   scaled dual updates.
%
% INPUT:
%   f              - 3D observed image volume (single precision recommended)
%   psf            - 3D point spread function, must match the size of f
%   bg             - 3D background estimate, same size as f
%   denoiser       - function handle for the denoiser, called as
%                    denoiser(image, denoiser_args)
%   denoiser_args  - struct of parameters passed to the denoiser
%   options        - (optional) struct with algorithm parameters:
%       .iter_num        - maximum number of ADMM iterations (default: 300)
%       .rho             - fallback ADMM penalty for rho1/rho2/rho3
%                          (default: 1)
%       .rho1            - ADMM penalty for Poisson data-fidelity constraint
%                          z1 = Hx + b (default: options.rho)
%       .rho2            - ADMM penalty for denoiser coupling constraint
%                          z2 = x (default: options.rho)
%       .rho3            - ADMM penalty for framelet sparsity constraint
%                          z3 = Wx (default: options.rho)
%       .alpha           - trade-off weight between deblurred and denoised
%                          images (default: 2e-4)
%       .beita           - framelet sparsity regularization weight
%                          (default: 1e-4)
%       .relaxation      - over-relaxation factor omega in (0, 2)
%                          (default: options.omega or 1.7)
%       .omega           - alias for .relaxation
%       .min_iter        - minimum iterations before early stopping
%                          (default: 2)
%       .track_residuals - whether to store true primal residual norms
%                          (default: false)
%       .stop_criterion  - relative error threshold for early stopping
%                          (default: 2e-5)
%       .frame           - framelet type following the 2D convention
%                          (1: Haar, 2: piecewise linear, 3: piecewise cubic)
%                          (default: 2)
%       .Level           - framelet decomposition levels (default: 4)
%       .useGPU          - whether to use GPU acceleration (default: 1)
%       .verbose         - whether to print iteration info (default: true)
%       .progress_fcn    - optional function handle called each iteration as
%                          progress_fcn(iter, iter_num, NSDE); empty = unused
%
% OUTPUT:
%   x           - deconvolved image volume, non-negative and normalized to [0, 1]
%   RelErr      - vector of relative error (NSDE) at each iteration
%   diagnostics - optional struct with relaxation, iteration count, elapsed
%                 time, stop flag, and optional primal residual history

%% Parse options with defaults
if nargin < 6, options = struct(); end

iter_num        = getOpt(options, 'iter_num',        300);
rho             = getOpt(options, 'rho',             1);
rho1            = getOpt(options, 'rho1',            rho);
rho2            = getOpt(options, 'rho2',            rho);
rho3            = getOpt(options, 'rho3',            rho);
alpha           = getOpt(options, 'alpha',           2e-4);
beita           = getOpt(options, 'beita',           1e-4);
relaxation      = getOpt(options, 'relaxation',      getOpt(options, 'omega', 1.7));
min_iter        = getOpt(options, 'min_iter',        2);
track_residuals = getOpt(options, 'track_residuals', false);
stop_criterion  = getOpt(options, 'stop_criterion',  2e-5);
frame           = getOpt(options, 'frame',           2);
Level           = getOpt(options, 'Level',           4);
useGPU          = getOpt(options, 'useGPU',          1);
verbose         = getOpt(options, 'verbose',         true);
progress_fcn    = getOpt(options, 'progress_fcn',    []);

if ~isscalar(relaxation) || ~isfinite(relaxation) || relaxation <= 0 || relaxation >= 2
    error('options.relaxation must be a finite scalar in the open interval (0, 2).');
end

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
if track_residuals
    primal_residual = double(zeros(iter_num, 1));
else
    primal_residual = [];
end

stopped = false;

%% ADMM iterations
tic
% Cache the forward predictions so the next iteration can reuse them.
Ax_new = real(ifftn(OTF .* fftn(x)));
Wx_new = W(x);

for iter = 1:iter_num
    if iter > 1
        x_old = x;
    end

    % --- Evaluate previous constraint predictions ---
    Ax_old = Ax_new;
    Wx_old = Wx_new;

    % --- Update z1 (Poisson data-fidelity proximal) ---
    t1 = Ax_old + bg + v(:,:,:,1) - 1./rho1;
    t2 = sqrt(max(t1.^2 + 4.*f./rho1, 0));
    z(:,:,:,1) = 0.5 .* (t1 + t2);

    % --- Update z2 (denoiser coupling proximal) ---
    z(:,:,:,2) = (alpha.*g + rho2.*(x + v(:,:,:,2))) ./ (alpha + rho2);

    % --- Update z3 (framelet sparsity proximal) ---
    z_3 = ShrinkFramelet3D(AddFrameletArray3D(Wx_old, v_3), beita./rho3);

    % --- Build over-relaxed auxiliary variables ---
    z1_hat = relaxation.*z(:,:,:,1) + (1 - relaxation).*(Ax_old + bg);
    z2_hat = relaxation.*z(:,:,:,2) + (1 - relaxation).*x;
    z3_hat = ScaleAddFrameletArray3D(relaxation, z_3, 1 - relaxation, Wx_old);

    % --- Update x (linear system solve in Fourier domain) ---
    WtW = WT(SubFrameletArray3D(z3_hat, v_3));
    XtX = rho1.*cOTF.*fftn(z1_hat - v(:,:,:,1) - bg) ...
        + rho2.*fftn(z2_hat - v(:,:,:,2)) ...
        + rho3.*fftn(WtW);
    x = single(real(ifftn(XtX ./ AtA)));

    % --- Evaluate latest constraint predictions ---
    Ax_new = real(ifftn(OTF .* fftn(x)));
    Wx_new = W(x);

    % --- Update dual variables v1, v2 ---
    v(:,:,:,1) = v(:,:,:,1) + Ax_new + bg - z1_hat;
    v(:,:,:,2) = v(:,:,:,2) + x - z2_hat;

    % --- Update dual variable v3 ---
    v_3 = SubFrameletArray3D(AddFrameletArray3D(Wx_new, v_3), z3_hat);

    if track_residuals
        r1 = Ax_new + bg - z1_hat;
        r2 = x - z2_hat;
        r3 = SubFrameletArray3D(Wx_new, z3_hat);
        rnorm = sqrt(norm(r1(:), 'fro').^2 + norm(r2(:), 'fro').^2 + FrameletNorm3D(r3).^2);
        if useGPU
            rnorm = gather(rnorm);
        end
        primal_residual(iter) = double(rnorm);
    end

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

    if ~isempty(progress_fcn)
        progress_fcn(iter, iter_num, NSDE);
    end

    if iter >= min_iter && iter > 1 && abs(NSDE) <= stop_criterion
        stopped = true;
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
if track_residuals
    primal_residual = primal_residual(1:iter);
end

if useGPU
    x = gather(max(0, x));
else
    x = max(0, x);
end
x = x ./ max(x(:));

if nargout > 2
    diagnostics = struct();
    diagnostics.relaxation = relaxation;
    diagnostics.iterations = iter;
    diagnostics.elapsed_time = elapsed;
    diagnostics.stopped = stopped;
    diagnostics.primal_residual = primal_residual;
end

end

%% ======================== Local helpers ========================
function val = getOpt(s, field, default)
% Retrieve a field from struct s, or return the default value.
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end

function R = ScaleAddFrameletArray3D(a, A, b, B)
% Compute a*A + b*B for matching multi-level framelet coefficient trees.
    if ~iscell(A) || ~iscell(B) || numel(A) ~= numel(B)
        error('A and B must be cell arrays with the same number of levels.');
    end

    R = cell(size(A));
    for ell = 1:numel(A)
        if ~isequal(size(A{ell}), size(B{ell}))
            error('A{ell} and B{ell} must have the same subband layout at every level.');
        end

        R{ell} = cell(size(A{ell}));
        for idx = 1:numel(A{ell})
            R{ell}{idx} = a.*A{ell}{idx} + b.*B{ell}{idx};
        end
    end
end

function val = FrameletNorm3D(A)
% Compute the Euclidean norm of a multi-level framelet coefficient tree.
    if ~iscell(A)
        error('A must be a framelet coefficient cell array.');
    end

    val = 0;
    for ell = 1:numel(A)
        for idx = 1:numel(A{ell})
            coeff = A{ell}{idx};
            val = val + norm(coeff(:), 'fro').^2;
        end
    end
    val = sqrt(val);
end
