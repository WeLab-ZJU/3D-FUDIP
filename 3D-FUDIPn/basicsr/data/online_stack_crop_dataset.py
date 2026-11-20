import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from tifffile import imread

from basicsr.data.paired_stack_dataset import _build_paths, pre_process
from basicsr.utils.registry import DATASET_REGISTRY


def normalize_percentile(volume, low, high):
    """Normalize a volume using percentile-based min-max scaling."""
    p_low, p_high = np.percentile(volume, (low, high))
    eps = 1e-10
    return ((volume - p_low) / (p_high - p_low + eps)).astype(np.float32, copy=False)


def _validate_patch_size(volume_shape, patch_size, lq_path, gt_path):
    z_size, y_size, x_size = volume_shape
    patch_z, patch_y, patch_x = patch_size
    if patch_z > z_size or patch_y > y_size or patch_x > x_size:
        raise ValueError(
            f'Patch size {patch_size} is larger than volume shape {volume_shape}. '
            f'LQ: {lq_path}. GT: {gt_path}')


def _to_channel_first(volume):
    """Convert (D, H, W) numpy array to (C, D, H, W) float tensor."""
    return torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(dim=0)


def _augment_pair(lq_tensor, gt_tensor, use_flip, use_rot90):
    """Apply paired 3D augmentation on (C, D, H, W) tensors."""
    return pre_process(lq_tensor, gt_tensor, use_flip=use_flip, use_rot90=use_rot90)


@DATASET_REGISTRY.register()
class OnlineMicroscopyCropDataset(Dataset):
    """Online random-crop dataset for paired 3D microscopy volumes.

    Each __getitem__ call randomly selects a volume pair and crops a patch with
    foreground/background filtering. Returned tensors have shape (C, D, H, W);
    after batching BasicSR receives (B, C, D, H, W).
    """

    def __init__(self, opt):
        super(OnlineMicroscopyCropDataset, self).__init__()
        self.opt = opt
        self.paths = _build_paths(opt)

        patch_size = opt.get('patch_size', opt.get('gt_size'))
        if patch_size is None:
            raise KeyError('OnlineMicroscopyCropDataset requires patch_size or gt_size.')
        self.patch_size = tuple(int(v) for v in patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f'patch_size must have 3 values (D, H, W), but got {self.patch_size}.')

        self.virtual_length = int(opt.get('virtual_length', 1000))
        self.percentile_low = float(opt.get('percentile_low', 0.01))
        self.percentile_high = float(opt.get('percentile_high', 99.99))
        self.intensity_threshold = float(opt.get('intensity_threshold', 0.11))
        self.area_ratio_threshold = float(opt.get('area_ratio_threshold', 0.008))
        self.raw_background_threshold = float(opt.get('raw_background_threshold', 0.15))
        self.max_sampling_attempts = int(opt.get('max_sampling_attempts', 512))
        self.cache_data = bool(opt.get('cache_data', True))
        self.use_flip = bool(opt.get('use_flip', True))
        self.use_rot90 = bool(opt.get('use_rot90', True))

        if not 0 <= self.area_ratio_threshold <= 1:
            raise ValueError('area_ratio_threshold must be between 0 and 1.')
        if not (0 <= self.percentile_low < self.percentile_high <= 100):
            raise ValueError(
                f'Percentiles must satisfy 0 <= low < high <= 100, but got '
                f'{self.percentile_low} and {self.percentile_high}.')

        self._area_threshold = self.area_ratio_threshold * np.prod(self.patch_size)
        self._volume_cache = {}
        self._validated_pairs = set()

        if self.cache_data:
            for i in range(len(self.paths)):
                self._load_normalized_pair(i)

    def __len__(self):
        return self.virtual_length

    def _load_normalized_pair(self, pair_idx):
        if self.cache_data and pair_idx in self._volume_cache:
            return self._volume_cache[pair_idx]

        pair = self.paths[pair_idx]
        lq_vol = imread(pair['lq_path']).astype(np.float32, copy=False)
        gt_vol = imread(pair['gt_path']).astype(np.float32, copy=False)

        if pair_idx not in self._validated_pairs:
            if lq_vol.ndim != 3 or gt_vol.ndim != 3:
                raise ValueError(
                    f'Expected 3D TIFF stacks (D, H, W). LQ {lq_vol.shape}: {pair["lq_path"]}. '
                    f'GT {gt_vol.shape}: {pair["gt_path"]}.')
            if lq_vol.shape != gt_vol.shape:
                raise ValueError(
                    f'LQ and GT shapes differ: {lq_vol.shape} vs {gt_vol.shape}. '
                    f'LQ: {pair["lq_path"]}. GT: {pair["gt_path"]}.')
            _validate_patch_size(lq_vol.shape, self.patch_size, pair['lq_path'], pair['gt_path'])
            self._validated_pairs.add(pair_idx)

        lq_vol = normalize_percentile(lq_vol, self.percentile_low, self.percentile_high)
        gt_vol = normalize_percentile(gt_vol, self.percentile_low, self.percentile_high)

        if self.cache_data:
            self._volume_cache[pair_idx] = (lq_vol, gt_vol, pair['lq_path'], pair['gt_path'])

        return lq_vol, gt_vol, pair['lq_path'], pair['gt_path']

    def _sample_valid_crop(self, lq_vol, gt_vol):
        z_size, y_size, x_size = lq_vol.shape
        patch_z, patch_y, patch_x = self.patch_size

        last_lq_crop = None
        last_gt_crop = None

        for _ in range(self.max_sampling_attempts):
            zt = random.randint(0, z_size - patch_z)
            yt = random.randint(0, y_size - patch_y)
            xt = random.randint(0, x_size - patch_x)

            lq_crop = lq_vol[zt:zt + patch_z, yt:yt + patch_y, xt:xt + patch_x]
            gt_crop = gt_vol[zt:zt + patch_z, yt:yt + patch_y, xt:xt + patch_x]
            last_lq_crop = lq_crop
            last_gt_crop = gt_crop

            if np.max(lq_crop) < self.raw_background_threshold:
                continue
            if np.count_nonzero(gt_crop > self.intensity_threshold) <= self._area_threshold:
                continue

            return lq_crop, gt_crop

        warnings.warn(
            f'Failed to sample a valid patch after {self.max_sampling_attempts} attempts.',
            RuntimeWarning,
            stacklevel=2)
        return last_lq_crop, last_gt_crop

    def __getitem__(self, idx):
        pair_idx = random.randint(0, len(self.paths) - 1)
        lq_vol, gt_vol, lq_path, gt_path = self._load_normalized_pair(pair_idx)

        lq_crop, gt_crop = self._sample_valid_crop(lq_vol, gt_vol)

        lq_tensor = _to_channel_first(lq_crop)
        gt_tensor = _to_channel_first(gt_crop)

        if self.opt['phase'] == 'train':
            lq_tensor, gt_tensor = _augment_pair(
                lq_tensor, gt_tensor, self.use_flip, self.use_rot90)

        return {
            'lq': lq_tensor,
            'gt': gt_tensor,
            'lq_path': lq_path,
            'gt_path': gt_path,
        }
