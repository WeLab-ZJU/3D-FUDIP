"""3D PSNR/SSIM metrics for BasicSR validation on (B, C, D, H, W) tensors."""

import warnings

import torch
import torch.nn.functional as F

from basicsr.utils.registry import METRIC_REGISTRY


def _ensure_batched_tensor(tensor, name):
    """Ensure input is a torch.Tensor with batch dimension."""
    if not torch.is_tensor(tensor):
        raise TypeError(f'{name} must be a torch.Tensor, but got {type(tensor)}.')
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 5:
        raise ValueError(
            f'{name} must have shape (B, C, D, H, W) or (C, D, H, W), but got {tensor.shape}.')
    return tensor


def _apply_crop_border_3d(tensor, crop_border):
    if crop_border == 0:
        return tensor
    return tensor[..., crop_border:-crop_border, crop_border:-crop_border, crop_border:-crop_border]


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss /= gauss.sum()
    return gauss.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(input_tensor, window):
    if input_tensor.dim() == 4:
        conv = F.conv2d
    elif input_tensor.dim() == 5:
        conv = F.conv3d
    else:
        raise ValueError(f'Expected a 4D or 5D tensor, but got shape {input_tensor.shape}.')

    channels = input_tensor.shape[1]
    output = input_tensor
    for dim_idx, size in enumerate(input_tensor.shape[2:]):
        if size >= window.shape[-1]:
            output = conv(output, weight=window.transpose(2 + dim_idx, -1), stride=1, padding=0, groups=channels)
        else:
            warnings.warn(
                f'Skipping Gaussian smoothing at spatial dim {dim_idx} for input shape '
                f'{input_tensor.shape} and window size {window.shape[-1]}.')
    return output


def _ssim_3d_core(pred, target, data_range=1.0, win_size=11, win_sigma=1.5, k=(0.01, 0.03)):
    """Compute 3D SSIM; assumes inputs are in [0, data_range]."""
    if pred.shape != target.shape:
        raise ValueError(f'Input tensors should have the same shape, but got {pred.shape} and {target.shape}.')
    if pred.dim() not in (4, 5):
        raise ValueError(f'Input tensors should be 4D or 5D, but got shape {pred.shape}.')
    if win_size % 2 != 1:
        raise ValueError('Window size should be odd.')

    window = _fspecial_gauss_1d(win_size, win_sigma).to(device=pred.device, dtype=pred.dtype)
    window = window.repeat([pred.shape[1]] + [1] * (pred.dim() - 1))

    k1, k2 = k
    c1 = (k1 * data_range)**2
    c2 = (k2 * data_range)**2

    mu1 = _gaussian_filter(pred, window)
    mu2 = _gaussian_filter(target, window)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _gaussian_filter(pred * pred, window) - mu1_sq
    sigma2_sq = _gaussian_filter(target * target, window) - mu2_sq
    sigma12 = _gaussian_filter(pred * target, window) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return torch.flatten(ssim_map, 2).mean(-1).mean()


def _legacy_psnr_tensor(pred, target, data_range=1.0):
    """Match original utils.metric_utils.psnr with explicit data_range."""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    mse = torch.mean((pred_flat - target_flat)**2)
    if mse == 0:
        return float('inf')
    return float((10.0 * torch.log10((data_range**2) / mse)).item())


@METRIC_REGISTRY.register()
def calculate_psnr_3d(img, img2, data_range=1.0, crop_border=0, **kwargs):
    """Calculate PSNR for 3D volumes.

    Legacy semantics: no max-normalization; PSNR uses MSE over all voxels with
    peak value ``data_range`` (default 1.0).

    Args:
        img (Tensor): Predicted volume, shape (B, C, D, H, W) or (C, D, H, W).
        img2 (Tensor): Ground-truth volume, same shape as img.
        data_range (float): Peak signal value used in PSNR formula.
        crop_border (int): Cropped voxels on each spatial edge (D/H/W).

    Returns:
        float: PSNR in dB.
    """
    del kwargs
    img = _ensure_batched_tensor(img, 'img')
    img2 = _ensure_batched_tensor(img2, 'img2')
    if img.shape != img2.shape:
        raise ValueError(f'Volume shapes are different: {img.shape}, {img2.shape}.')

    img = _apply_crop_border_3d(img, crop_border)
    img2 = _apply_crop_border_3d(img2, crop_border)

    values = []
    for batch_idx in range(img.shape[0]):
        values.append(_legacy_psnr_tensor(img[batch_idx], img2[batch_idx], data_range=data_range))
    return float(sum(values) / len(values))


@METRIC_REGISTRY.register()
def calculate_ssim_3d(img, img2, data_range=1.0, win_size=11, win_sigma=1.5, crop_border=0, **kwargs):
    """Calculate SSIM for 3D volumes.

    No max-normalization; SSIM uses ``data_range`` (default 1.0) as the peak
    signal value for the stability constants ``c1``/``c2``.

    Args:
        img (Tensor): Predicted volume, shape (B, C, D, H, W) or (C, D, H, W).
        img2 (Tensor): Ground-truth volume, same shape as img.
        data_range (float): Peak signal value used in SSIM constants.
        win_size (int): Gaussian window size (odd).
        win_sigma (float): Gaussian window sigma.
        crop_border (int): Cropped voxels on each spatial edge (D/H/W).

    Returns:
        float: SSIM.
    """
    del kwargs
    img = _ensure_batched_tensor(img, 'img')
    img2 = _ensure_batched_tensor(img2, 'img2')
    if img.shape != img2.shape:
        raise ValueError(f'Volume shapes are different: {img.shape}, {img2.shape}.')

    img = _apply_crop_border_3d(img, crop_border)
    img2 = _apply_crop_border_3d(img2, crop_border)

    values = []
    for batch_idx in range(img.shape[0]):
        pred = img[batch_idx]
        target = img2[batch_idx]
        ssim_value = _ssim_3d_core(
            pred, target, data_range=data_range, win_size=win_size, win_sigma=win_sigma)
        values.append(float(ssim_value.item()))
    return float(sum(values) / len(values))
