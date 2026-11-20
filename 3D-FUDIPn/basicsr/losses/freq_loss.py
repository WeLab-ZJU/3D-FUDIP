import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY

# version adaptation for PyTorch > 1.7.1
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
if IS_HIGH_VERSION:
    import torch.fft


def fft_spectrum(x, gamma=0.1, eps=1e-8):
    """Return compressed magnitude spectrum for 3D tensors without apodization."""
    if not IS_HIGH_VERSION:
        raise RuntimeError('loss_type="fft_mse" requires torch.fft support (PyTorch > 1.7.1).')
    fft = torch.fft.fftn(torch.complex(x, torch.zeros_like(x)), dim=(-3, -2, -1))
    absfft = torch.pow(torch.abs(fft) + eps, gamma)
    return torch.fft.fftshift(absfft, dim=(-3, -2, -1))

class FocalFreqLoss3D(nn.Module):
    """The torch.nn.Module class that implements the 3D focal frequency loss - a
    frequency domain loss function for optimizing models on volumetric (N, C, D, H, W) data.

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    https://arxiv.org/pdf/2012.12821.pdf

    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFreqLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        """
        Transforms a 3D spatial tensor into its frequency domain representation via 3D DFT.

        Args:
            x (torch.Tensor): Input spatial tensor of shape (N, C, D, H, W).
            
        Returns:
            torch.Tensor: Frequency representation of shape (N, patch_factor**3, C, D_patch, H_patch, W_patch, 2).
        """
        # crop volume patches
        patch_factor = self.patch_factor
        _, _, d, h, w = x.shape
        assert d % patch_factor == 0 and h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by volume depth, height and width')
        patch_list = []
        patch_d = d // patch_factor
        patch_h = h // patch_factor
        patch_w = w // patch_factor

        # Extract 3D local patches
        for i in range(patch_factor):
            for j in range(patch_factor):
                for k in range(patch_factor):
                    patch_list.append(x[:, :,
                                        i * patch_d:(i + 1) * patch_d,
                                        j * patch_h:(j + 1) * patch_h,
                                        k * patch_w:(k + 1) * patch_w])

        # stack to patch tensor -> shape (N, P, C, D, H, W)
        y = torch.stack(patch_list, 1)

        # Perform 3D DFT over the last three spatial dims (real-to-complex, orthonormalization)
        if IS_HIGH_VERSION:
            freq = torch.fft.fftn(y, dim=(-3, -2, -1), norm='ortho')
            freq = torch.stack([freq.real, freq.imag], -1)
        else:
            freq = torch.rfft(y, 3, onesided=False, normalized=True)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        """
        Formulates and calculates the weighted frequency domain loss.

        Args:
            recon_freq (torch.Tensor): Reconstructed frequency tensor.
            real_freq (torch.Tensor): Target/Real frequency tensor.
            matrix (torch.Tensor, optional): Pre-calculated spectrum weight matrix. Default: None.
            
        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # Apply logarithmic scaling if configured
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # Normalize the weight matrix
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                # Flatten spatial dimensions (Depth, Height, Width) to compute local maximum safely
                max_values = matrix_tmp.flatten(start_dim=-3).max(dim=-1)[0]
                # Broadcast the max values back to spatial dimensions
                matrix_tmp = matrix_tmp / max_values[..., None, None, None]
            
            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.

        Args:
            pred (torch.Tensor): of shape (N, C, D, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, D, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight


@LOSS_REGISTRY.register()
class FreqLoss3D(nn.Module):
    """Frequency-domain loss for volumetric tensors of shape (B, C, D, H, W)."""

    def __init__(self, loss_weight=1.0, loss_type='focal', alpha=1.0, patch_factor=1,
                 ave_spectrum=False, log_matrix=False, batch_matrix=False, gamma=0.1, eps=1e-8):
        super().__init__()
        loss_type = loss_type.lower()
        if loss_type not in ('focal', 'fft_mse'):
            raise ValueError(f'Unsupported loss_type: {loss_type!r}. Supported values are: focal, fft_mse.')
        self.loss_type = loss_type
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.eps = eps
        if self.loss_type == 'focal':
            self.focal_loss = FocalFreqLoss3D(
                loss_weight=loss_weight,
                alpha=alpha,
                patch_factor=patch_factor,
                ave_spectrum=ave_spectrum,
                log_matrix=log_matrix,
                batch_matrix=batch_matrix)

    def _fft_mse(self, pred, target):
        """Calculate MSE between compressed 3D FFT magnitude spectra."""
        if pred.shape != target.shape:
            raise ValueError(f'pred and target must have the same shape, got {pred.shape} and {target.shape}')
        if pred.ndim != 5:
            raise ValueError(
                f'pred and target must be 5D tensors of shape (B, C, D, H, W), got {pred.ndim}D')

        pred_fft = fft_spectrum(pred, gamma=self.gamma, eps=self.eps)
        with torch.no_grad():
            target_fft = fft_spectrum(target, gamma=self.gamma, eps=self.eps)
        return torch.mean((target_fft - pred_fft) ** 2)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function for frequency-domain loss."""
        if self.loss_type == 'focal':
            return self.focal_loss(pred, target, matrix=matrix, **kwargs)
        return self._fft_mse(pred, target) * self.loss_weight