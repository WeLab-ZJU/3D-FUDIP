import random
from glob import glob
from os import path as osp
import numpy as np
import numexpr
import torch
from torch.utils.data import Dataset
from tifffile import imread

from basicsr.utils.registry import DATASET_REGISTRY


def normalize_min_max(im, max_v, min_v=0):
    """Normalize an array with explicit lower and upper intensity bounds."""
    eps = 1e-10
    im = (im - min_v) / (max_v - min_v + eps)
    return im.astype(np.float32, copy=False)


def _sorted_glob(pattern):
    paths = sorted(glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f'No TIFF stack files found with pattern: {pattern}')
    return paths


def _scan_stack_folder(folder):
    suffixes = ('*.tif', '*.tiff', '*.TIF', '*.TIFF')
    paths = []
    for suffix in suffixes:
        paths.extend(glob(osp.join(folder, suffix)))
    paths = sorted(set(paths))
    if len(paths) == 0:
        raise FileNotFoundError(f'No TIFF stack files found in folder: {folder}')
    return paths


def _pair_sorted_paths(lq_paths, gt_paths):
    if len(lq_paths) != len(gt_paths):
        raise ValueError(
            f'LQ and GT datasets have different numbers of stacks: {len(lq_paths)} and {len(gt_paths)}.')
    return [dict(lq_path=lq_path, gt_path=gt_path) for lq_path, gt_path in zip(lq_paths, gt_paths)]


def _paths_from_root_and_prefix(root_path, gt_prefix, lq_prefix):
    gt_pattern = osp.join(root_path, gt_prefix)
    lq_pattern = osp.join(root_path, lq_prefix)
    gt_paths = _sorted_glob(gt_pattern)
    lq_paths = _sorted_glob(lq_pattern)
    return _pair_sorted_paths(lq_paths, gt_paths)


def _paths_from_folders(gt_folder, lq_folder, filename_tmpl='{}'):
    gt_paths = _scan_stack_folder(gt_folder)
    lq_paths = _scan_stack_folder(lq_folder)

    if filename_tmpl != '{}':
        lq_path_map = {osp.basename(path): path for path in lq_paths}
        paired_paths = []
        for gt_path in gt_paths:
            basename, ext = osp.splitext(osp.basename(gt_path))
            lq_name = f'{filename_tmpl.format(basename)}{ext}'
            if lq_name not in lq_path_map:
                raise FileNotFoundError(f'{lq_name} is not found in LQ folder: {lq_folder}')
            paired_paths.append(dict(lq_path=lq_path_map[lq_name], gt_path=gt_path))
        return paired_paths

    return _pair_sorted_paths(lq_paths, gt_paths)


def _build_paths(opt):
    root_path = opt.get('root_path')
    gt_prefix = opt.get('gt_prefix', opt.get('hq_prefix'))
    lq_prefix = opt.get('lq_prefix', opt.get('low_prefix'))
    if root_path is not None and gt_prefix is not None and lq_prefix is not None:
        return _paths_from_root_and_prefix(root_path, gt_prefix, lq_prefix)

    gt_folder = opt.get('dataroot_gt')
    lq_folder = opt.get('dataroot_lq')
    if gt_folder is not None and lq_folder is not None:
        filename_tmpl = opt.get('filename_tmpl', '{}')
        return _paths_from_folders(gt_folder, lq_folder, filename_tmpl)

    raise KeyError('MicroscopyDataset requires either root_path with gt_prefix/lq_prefix or '
                   'dataroot_gt/dataroot_lq.')


def rand_flag():
    return torch.rand(1).item() > 0.5


def pre_process(img1, img2, use_flip=True, use_rot90=True):
    """Apply paired 3D augmentation on (C, D, H, W) tensors."""
    if use_rot90 and img1.shape[-2] == img1.shape[-1]:
        k = random.randint(0, 3)
        if k > 0:
            img1 = torch.rot90(img1, k, dims=(-2, -1))
            img2 = torch.rot90(img2, k, dims=(-2, -1))
    if use_flip:
        if rand_flag():
            img1 = torch.flip(img1, [-3])
            img2 = torch.flip(img2, [-3])
        if rand_flag():
            img1 = torch.flip(img1, [-2])
            img2 = torch.flip(img2, [-2])
        if rand_flag():
            img1 = torch.flip(img1, [-1])
            img2 = torch.flip(img2, [-1])

    return img1, img2


@DATASET_REGISTRY.register()
class MicroscopyDataset(Dataset):
    """
    Dataset class for 3D optical microscopy or confocal data.
    LQ: Low Quality (blur/noisy data)
    GT: Ground truth/clear data.

    The returned single-sample tensors have shape (C, D, H, W). After
    batching with a DataLoader, BasicSR receives tensors with shape
    (B, C, D, H, W).
    """
    def __init__(self, opt):
        super(MicroscopyDataset, self).__init__()
        self.opt = opt
        self.paths = _build_paths(opt)
        self.normalize_min_max = bool(opt.get('normalize_min_max', False))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        gt_path = self.paths[idx]['gt_path']
        lq_path = self.paths[idx]['lq_path']

        lq_img = imread(lq_path)
        gt_img = imread(gt_path)

        lq_img = lq_img.astype('float32', copy=False)
        gt_img = gt_img.astype('float32', copy=False)

        if lq_img.ndim != 3:
            raise ValueError(
                f'LQ stack should be a 3D TIFF with shape (D, H, W), but got {lq_img.shape}: {lq_path}')
        if gt_img.ndim != 3:
            raise ValueError(
                f'GT stack should be a 3D TIFF with shape (D, H, W), but got {gt_img.shape}: {gt_path}')

        if self.normalize_min_max:
            lq_img = normalize_min_max(lq_img, max_v=lq_img.max(), min_v=lq_img.min())
            gt_img = normalize_min_max(gt_img, max_v=gt_img.max(), min_v=gt_img.min())

        lq_img = torch.from_numpy(np.ascontiguousarray(lq_img)).unsqueeze(dim=0)
        gt_img = torch.from_numpy(np.ascontiguousarray(gt_img)).unsqueeze(dim=0)

        # Single-sample tensor shape: (channels, depth, height, width).
        if self.opt['phase'] == 'train':
            use_flip = bool(self.opt.get('use_flip', True))
            use_rot90 = bool(self.opt.get('use_rot90', True))
            lq_img, gt_img = pre_process(lq_img, gt_img, use_flip=use_flip, use_rot90=use_rot90)

        return {'lq': lq_img, 'gt': gt_img, 'lq_path': lq_path, 'gt_path': gt_path}
