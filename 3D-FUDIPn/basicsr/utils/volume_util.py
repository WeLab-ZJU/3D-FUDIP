"""Utilities for 3D volume tensors (B, C, D, H, W) used in BasicSR 3D pipelines."""

import os

import numpy as np
import torch
from tifffile import imwrite as tiff_imwrite


def _ensure_batched_volume(tensor, name='tensor'):
    """Ensure input is a 5D volume tensor (B, C, D, H, W)."""
    if not torch.is_tensor(tensor):
        raise TypeError(f'{name} must be a torch.Tensor, but got {type(tensor)}.')
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 5:
        raise ValueError(
            f'{name} must have shape (B, C, D, H, W) or (C, D, H, W), but got {tuple(tensor.shape)}.')
    return tensor


def tensor2volume(tensor, channel=0):
    """Convert a 3D volume tensor to a list of NumPy arrays.

    Args:
        tensor (Tensor): Shape (B, C, D, H, W) or (C, D, H, W).
        channel (int): Channel index used when exporting single-channel volumes.

    Returns:
        list[ndarray]: Each element has shape (D, H, W) with float32 dtype.
    """
    tensor = _ensure_batched_volume(tensor)
    if channel < 0 or channel >= tensor.shape[1]:
        raise ValueError(f'channel={channel} is out of range for tensor with C={tensor.shape[1]}.')

    volumes = []
    for batch_idx in range(tensor.shape[0]):
        volume = tensor[batch_idx, channel].detach().float().cpu().numpy()
        volumes.append(np.ascontiguousarray(volume, dtype=np.float32))
    return volumes


def volume2preview(tensor, mode='mip'):
    """Project a 3D volume tensor to a 2D preview tensor for visualization.

    Args:
        tensor (Tensor): Shape (B, C, D, H, W) or (C, D, H, W).
        mode (str): Projection mode. Only ``mip`` (max intensity projection) is supported.

    Returns:
        Tensor: Shape (B, C, H, W) or (C, H, W), detached on CPU.
    """
    if mode != 'mip':
        raise ValueError(f'Unsupported preview mode: {mode}. Only "mip" is supported.')

    has_batch = tensor.dim() == 5
    tensor = _ensure_batched_volume(tensor)
    preview = tensor.max(dim=2).values.detach().cpu()
    if not has_batch:
        preview = preview.squeeze(0)
    return preview


def imwrite_volume(volume, file_path, auto_mkdir=True):
    """Write a single 3D volume to a TIFF stack.

    Args:
        volume (ndarray): Volume with shape (D, H, W).
        file_path (str): Output TIFF path.
        auto_mkdir (bool): Create parent directories when missing.

    Returns:
        None
    """
    if not isinstance(volume, np.ndarray):
        raise TypeError(f'volume must be a numpy.ndarray, but got {type(volume)}.')
    if volume.ndim != 3:
        raise ValueError(f'volume must have shape (D, H, W), but got {volume.shape}.')

    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)

    tiff_imwrite(file_path, volume.astype(np.float32, copy=False))
