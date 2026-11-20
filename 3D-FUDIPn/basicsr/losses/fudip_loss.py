import warnings

import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY


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


def _ssim_3d(pred, target, data_range=1.0, win_size=11, win_sigma=1.5, k=(0.01, 0.03)):
    if pred.shape != target.shape:
        raise ValueError(f'Input tensors should have the same shape, but got {pred.shape} and {target.shape}.')
    if pred.dim() not in (4, 5):
        raise ValueError(f'Input tensors should be 4D or 5D, but got shape {pred.shape}.')
    if pred.dtype != target.dtype:
        raise ValueError(f'Input tensors should have the same dtype, but got {pred.dtype} and {target.dtype}.')
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


@LOSS_REGISTRY.register()
class SSIMLoss3D(nn.Module):
    """SSIM-based loss for 3D FUDIP training tensors.

    SSIM is computed on the raw tensors using ``data_range`` (default 1.0) and
    mapped to a minimization objective via ``-log((1 + ssim) / 2)``.
    """

    def __init__(self, loss_weight=1.0, data_range=1.0, win_size=11, win_sigma=1.5):
        super(SSIMLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma

    def forward(self, pred, target):
        ssim_value = _ssim_3d(
            pred, target, data_range=self.data_range, win_size=self.win_size, win_sigma=self.win_sigma)
        return self.loss_weight * (-torch.log((1 + ssim_value) / 2))
