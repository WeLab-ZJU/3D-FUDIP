"""
Refactored and optimized 3D Gradient Magnitude Edge Loss (GMELoss3D).
Adapted from https://github.com/AgamChopra/3D-Edge-Loss/blob/main/GradEdgeLoss3D.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from basicsr.utils.registry import LOSS_REGISTRY

from .basic_loss import CharbonnierLoss

_SUPPORTED_EDGE_LOSS_TYPES = ('l1', 'charbonnier')


def get_sobel_kernel3D(n1=1, n2=2, n3=2):
    """
    Constructs 3D Sobel kernels for spatial gradient extraction.

    Parameters
    ----------
    n1 : int
        Weight for the corner components.
    n2 : int
        Weight for the adjacent components.
    n3 : int
        Multiplier for the central axis components.

    Returns
    -------
    np.ndarray
        Array of shape (9, 3, 3, 3) containing 9 directional 3D kernels.

    Notes
    -----
    Each kernel is a 3x3x3 array whose numpy axes are ordered (z, y, x), i.e.
    axis-0 = z, axis-1 = y, axis-2 = x.  Stack index and gradient direction:

    Diagonal names use plane prefix + index (1/2 for the two diagonals
    in that plane): Sxy_1/2, Syz_1/2, Sxz_1/2.

    ======  =========  ==========================
    index   variable   direction (+x, +y, +z)
    ======  =========  ==========================
    0       Sx         (+x, 0, 0)
    1       Sy         (0, +y, 0)
    2       Sz         (0, 0, +z)
    3       Sxy_1      (+x, -y, 0)
    4       Sxy_2      (+x, +y, 0)
    5       Syz_1      (0, -y, +z)
    6       Syz_2      (0, +y, +z)
    7       Sxz_1      (+x, 0, -z)
    8       Sxz_2      (+x, 0, +z)
    ======  =========  ==========================
    """
    # Orthogonal kernels along +x, +y, +z
    Sx = np.array(
        [[[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]],
         [[-n2, 0, n2], [-n3*n2, 0, n3*n2], [-n2, 0, n2]],
         [[-n1, 0, n1], [-n2, 0, n2], [-n1, 0, n1]]])

    Sy = np.array(
        [[[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]],
         [[-n2, -n3*n2, -n2], [0, 0, 0], [n2, n3*n2, n2]],
         [[-n1, -n2, -n1], [0, 0, 0], [n1, n2, n1]]])

    Sz = np.array(
        [[[-n1, -n2, -n1], [-n2, -n3*n2, -n2], [-n1, -n2, -n1]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[n1, n2, n1], [n2, n3*n2, n2], [n1, n2, n1]]])

    # Diagonal kernels: S{plane}_1/2, two diagonals per coordinate plane
    Sxy_1 = np.array(
        [[[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]],
         [[0, n2, n2*n3], [-n2, 0, n2], [-n2*n3, -n2, 0]],
         [[0, n1, n2], [-n1, 0, n1], [-n2, -n1, 0]]])

    Sxy_2 = np.array(
        [[[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]],
         [[-n2*n3, -n2, 0], [-n2, 0, n2], [0, n2, n2*n3]],
         [[-n2, -n1, 0], [-n1, 0, n1], [0, n1, n2]]])

    Syz_1 = Sxy_1.T
    Syz_2 = Sxy_2.T
    Sxz_1 = np.array([-S.T for S in Sxy_1.T])
    Sxz_2 = np.array([S.T for S in Sxy_2.T])

    kernels = np.stack([
        Sx, Sy, Sz,
        Sxy_1, Sxy_2,
        Syz_1, Syz_2,
        Sxz_1, Sxz_2,
    ])
    return kernels


class GradEdge3D(nn.Module):
    """
    Computes the Gradient Magnitude of a 3D image tensor.
    Utilizes 9 directional Sobel filters efficiently using 3D convolutions.
    """
    def __init__(self, n1=1, n2=2, n3=2, normalize=False, eps=1e-6):
        """
        Parameters
        ----------
        n1, n2, n3 : int
            Sobel filter hyperparameters.
        normalize : bool
            If True, applies instance-wise min-max normalization. Generally
            discouraged for loss calculation to preserve absolute gradient scaling.
        eps : float
            Small constant for numerical stability in sqrt and normalization.
        """
        super().__init__()
        self.normalize = normalize
        self.eps = eps
        
        # Shape: (9, 3, 3, 3)
        kernels_np = get_sobel_kernel3D(n1, n2, n3)
        # Reshape to PyTorch Conv3d weight format: (out_channels, in_channels/groups, D, H, W)
        # Here, in_channels=1, so shape becomes (9, 1, 3, 3, 3)
        weight = torch.from_numpy(kernels_np).float().unsqueeze(1)
        
        # Register as buffer to ensure automatic device transfer alongside the model
        self.register_buffer('sobel_weight', weight)

    def forward(self, img):
        """
        Extracts structural edges.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor of shape (B, C, D, H, W). 

        Returns
        -------
        torch.Tensor
            Gradient magnitude tensor of shape (B, 1, D, H, W).
        """
        B, C, D, H, W = img.shape
        if min(D, H, W) <= 1:
            raise ValueError('Reflect padding with pad=1 requires D, H, W > 1.')

        # Reshape to process all channels uniformly in a single batch dimension
        # New shape: (B * C, 1, D, H, W)
        img_reshaped = img.reshape(B * C, 1, D, H, W)
        
        # Apply reflection padding manually to prevent boundary artifacts
        pad_size = 1
        pad_tuple = (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size)
        img_padded = F.pad(img_reshaped, pad_tuple, mode='reflect')
        
        # Perform 3D convolution with padding=0 since manual padding was applied
        # Output shape: (B * C, 9, D, H, W)
        weight = self.sobel_weight.to(device=img.device, dtype=img.dtype)
        gradients = F.conv3d(img_padded, weight, padding=0)
        
        # Calculate L2 norm of the gradients across the 9 directional kernels
        # Summing squares prevents cancellation of opposing gradients
        grad_mag_sq = torch.sum(gradients ** 2, dim=1)  # Shape: (B * C, D, H, W)
        grad_mag = torch.sqrt(grad_mag_sq + self.eps)
        
        # Reshape back to separate Batch and Channel dimensions
        grad_mag = grad_mag.view(B, C, D, H, W)
        
        # Average the gradient magnitudes across all original channels
        # Shape: (B, 1, D, H, W)
        grad_mag = grad_mag.mean(dim=1, keepdim=True)
        
        if self.normalize:
            # Instance-wise Min-Max Normalization (NOT batch-wise)
            # Flatten spatial dimensions to compute min/max per item in batch
            flat_grad = grad_mag.view(B, -1)
            grad_min = flat_grad.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
            grad_max = flat_grad.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1, 1)
            dnm = grad_max - grad_min + self.eps
            grad_mag = (grad_mag - grad_min) / dnm
            
        return grad_mag

def _build_edge_pixel_loss(loss_type):
    """Build the pixel-wise loss applied on extracted gradient magnitudes."""
    loss_type = loss_type.lower()
    if loss_type == 'l1':
        return nn.L1Loss()
    if loss_type == 'charbonnier':
        # Inner loss_weight is fixed to 1.0; GMELoss3D applies loss_weight globally.
        return CharbonnierLoss(loss_weight=1.0)
    raise ValueError(
        f'Unsupported loss_type: {loss_type!r}. '
        f'Supported values are: {_SUPPORTED_EDGE_LOSS_TYPES}.')


@LOSS_REGISTRY.register()
class EdgeLoss3D(nn.Module):
    """Gradient Magnitude Edge Loss for 3D tensors.

    Penalizes discrepancies in structural gradients between predictions and targets.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        loss_type (str): Pixel loss on gradient magnitudes. ``l1`` or ``charbonnier``.
            Default: ``l1``.
        n1, n2, n3 (int): Sobel filter hyperparameters.
        normalize_edge (bool): Whether to normalize extracted gradients before loss.
        eps (float): Numerical stability constant for GradEdge3D.
    """

    def __init__(self, loss_weight=1.0, loss_type='l1', n1=1, n2=2, n3=2,
                 normalize_edge=False, eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.edge_filter = GradEdge3D(n1, n2, n3, normalize=normalize_edge, eps=eps)
        self.pixel_loss = _build_edge_pixel_loss(loss_type)

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): Prediction tensor of shape (B, C, D, H, W).
            target (Tensor): Ground truth tensor of shape (B, C, D, H, W).
            weight (Tensor, optional): Unused. Kept for BasicSR loss API compatibility.
        """
        if pred.shape != target.shape:
            raise ValueError(
                f'pred and target must have the same shape, got {pred.shape} and {target.shape}')
        if pred.ndim != 5:
            raise ValueError(
                f'pred and target must be 5D tensors of shape (B, C, D, H, W), got {pred.ndim}D')

        # Gradients do not need to be backpropagated through the target tensor
        with torch.no_grad():
            edge_target = self.edge_filter(target)

        edge_pred = self.edge_filter(pred)
        if isinstance(self.pixel_loss, CharbonnierLoss):
            loss = self.pixel_loss(edge_pred, edge_target, weight=weight, **kwargs)
        else:
            loss = self.pixel_loss(edge_pred, edge_target)
        return self.loss_weight * loss

