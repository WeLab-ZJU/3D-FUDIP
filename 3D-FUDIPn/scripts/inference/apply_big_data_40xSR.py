"""Tiled 3D inference for large microscopy TIFF stacks via BasicSR.

Loads network architecture and weights from a train/test YAML, splits each
volume into overlapping patches, runs inference, and stitches the result.

Supports IDE one-click run (edit DEFAULT_* below) and CLI overrides via argparse.
"""

import argparse
import glob
import logging
import os
import re
import sys
from copy import deepcopy
from os import path as osp
from types import SimpleNamespace

import numpy as np
import torch
from tifffile import imread, imwrite


def get_repo_root():
    """Locate repo root by walking up until basicsr/ and options/ are found."""
    here = osp.abspath(osp.dirname(__file__))
    candidate = here
    for _ in range(6):
        if osp.isdir(osp.join(candidate, 'basicsr')) and osp.isdir(osp.join(candidate, 'options')):
            return candidate
        parent = osp.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    raise RuntimeError(f'Cannot locate repo root from script path: {here}')


ROOT = get_repo_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from basicsr.archs import build_network
from basicsr.utils import set_random_seed, yaml_load

# --- Edit these defaults for IDE one-click run ---
DEFAULT_OPT = 'options/test/FUDIP/test_FUDIP_2stage_40xSR.yml'
DEFAULT_CHECKPOINT = ''
DEFAULT_DATA_FOLDER = 'E:/FUDIP_data/deep-tissue/cropped_subvolome/upsampled_subvolume/'
DEFAULT_OUTPUT_DIR = 'E:/FUDIP_data/deep-tissue/cropped_subvolome/upsampled_subvolume/FUDIP_output_v5_fft/'
DEFAULT_PARAM_KEY = 'params_ema' # 'params' or 'params_ema'
DEFAULT_GPU = 0
DEFAULT_STRICT_LOAD = True
DEFAULT_IMG_W = 128
DEFAULT_IMG_H = 128
DEFAULT_IMG_S = 64
DEFAULT_GAP_W = 112
DEFAULT_GAP_H = 112
DEFAULT_GAP_S = 54
DEFAULT_BOUNDARY_PIXEL = 0
DEFAULT_CUT_PIXEL = 1
DEFAULT_START_NUM = 0
DEFAULT_END_NUM = 100
DEFAULT_XY_RESOLUTION = 0.05415
DEFAULT_Z_SPACING = 0.240
DEFAULT_PERCENTILE_LOW = 0.01
DEFAULT_PERCENTILE_HIGH = 99.99
DEFAULT_PATCH_MIN_MAX = True
# -------------------------------------------------


def resolve_path(path, base_dir=ROOT):
    """Resolve a path relative to repo root when not absolute."""
    if path is None:
        return None
    path = osp.expanduser(path)
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(base_dir, path))


def natural_sort_key(file_name):
    """Natural sort key: compare numeric substrings as integers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', file_name)]


def time_series_sort_key(file_name):
    """Sort time-series stacks by T index in the filename."""
    match = re.search(r'T(\d+)', file_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    raise ValueError(f'The file name is not a time series: {file_name}')


def scan_flat_tiff_paths(data_folder, sort_by_time=False):
    """Collect sorted 3D TIFF paths from a flat directory."""
    if not osp.isdir(data_folder):
        raise FileNotFoundError(f'Data folder not found: {data_folder}')

    suffixes = ('*.tif', '*.tiff', '*.TIF', '*.TIFF')
    paths = []
    for suffix in suffixes:
        paths.extend(glob.glob(osp.join(data_folder, suffix)))
    paths = list(set(paths))
    if sort_by_time:
        paths.sort(key=lambda p: time_series_sort_key(osp.basename(p)))
    else:
        paths.sort(key=lambda p: natural_sort_key(osp.basename(p)))
    if len(paths) == 0:
        raise FileNotFoundError(f'No TIFF stack files found in: {data_folder}')
    return paths


def load_opt(opt_path):
    """Load and validate option YAML as a mapping."""
    if not osp.isfile(opt_path):
        raise FileNotFoundError(
            f'Option YAML not found: {opt_path}\n'
            f'Repo root resolved to: {ROOT}\n'
            'Check --opt path or edit DEFAULT_OPT at the top of this script.')
    opt = yaml_load(opt_path)
    if not isinstance(opt, dict):
        raise TypeError(f'Option YAML must be a mapping, got {type(opt).__name__}: {opt_path}')
    return opt


def find_latest_checkpoint(opt_path):
    """Pick the latest net_g_*.pth under experiments/{name}/models/."""
    opt = load_opt(opt_path)
    exp_name = opt.get('name')
    if exp_name is None:
        raise KeyError(f'Missing "name" in option file: {opt_path}')

    models_dir = resolve_path(osp.join('experiments', exp_name, 'models'))
    pattern = osp.join(models_dir, 'net_g_*.pth')
    checkpoints = sorted(glob.glob(pattern))
    if len(checkpoints) == 0:
        raise FileNotFoundError(f'No checkpoints matched: {pattern}')
    return checkpoints[-1]


def load_network_weights(net, load_path, strict=True, param_key='params_ema'):
    """Load checkpoint weights with BasicSR-compatible key handling."""
    logger = logging.getLogger('basicsr')
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)

    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            logger.info('params_ema does not exist, fallback to params.')
            param_key = 'params'
        if param_key not in load_net:
            raise KeyError(f'Neither "{param_key}" nor "params" found in checkpoint: {load_path}')
        load_net = load_net[param_key]

    for key, value in list(deepcopy(load_net).items()):
        if key.startswith('module.'):
            load_net[key[7:]] = value
            load_net.pop(key)

    logger.info('Loading %s from %s with param key [%s].', net.__class__.__name__, load_path, param_key)
    net.load_state_dict(load_net, strict=strict)


def patch_size_from_yaml(opt):
    """Read datasets.train.patch_size [D, H, W] when present."""
    train_ds = opt.get('datasets', {}).get('train', {})
    patch_size = train_ds.get('patch_size')
    if not patch_size or len(patch_size) != 3:
        return None
    img_s, img_h, img_w = patch_size
    return img_s, img_h, img_w


def _tile_attr(tile_opt, name):
    """Read a tiling option from Namespace or dict."""
    if isinstance(tile_opt, dict):
        return tile_opt[name]
    return getattr(tile_opt, name)


def build_tile_opt(args, yaml_opt):
    """Build tiling namespace, optionally filling patch size from YAML."""
    img_s, img_h, img_w = args.img_s, args.img_h, args.img_w
    defaults_changed = (
        args.img_w != DEFAULT_IMG_W or args.img_h != DEFAULT_IMG_H or args.img_s != DEFAULT_IMG_S)
    if not defaults_changed:
        yaml_patch = patch_size_from_yaml(yaml_opt)
        if yaml_patch is not None:
            img_s, img_h, img_w = yaml_patch

    return SimpleNamespace(
        img_w=img_w,
        img_h=img_h,
        img_s=img_s,
        gap_w=args.gap_w,
        gap_h=args.gap_h,
        gap_s=args.gap_s,
    )


def normalize_percentile(volume, percentile_low=0.01, percentile_high=99.99, eps=1e-10):
    """Clip a volume by percentiles and normalize it to [0, 1]."""
    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError(
            f'Percentiles must satisfy 0 <= low < high <= 100, but got '
            f'{percentile_low} and {percentile_high}.')

    volume = volume.astype(np.float32, copy=False)
    v_min = np.percentile(volume, percentile_low)
    v_max = np.percentile(volume, percentile_high)
    v_range = v_max - v_min
    if v_range <= eps:
        return np.zeros_like(volume, dtype=np.float32)

    volume = np.clip(volume, v_min, v_max)
    volume = (volume - v_min) / (v_range + eps)
    return volume.astype(np.float32, copy=False)


def normalize_patch_min_max(patch, eps=1e-10):
    """Clamp negative values and min-max normalize one inference patch."""
    patch = patch.astype(np.float32, copy=True)
    patch[patch < 0] = 0

    patch_min = float(patch.min())
    patch_max = float(patch.max())
    patch_range = patch_max - patch_min

    if patch_range <= eps:
        patch_norm = np.zeros_like(patch, dtype=np.float32)
    else:
        patch_norm = (patch - patch_min) / (patch_range + eps)
        patch_norm = patch_norm.astype(np.float32, copy=False)

    meta = {
        'patch_min': patch_min,
        'patch_max': patch_max,
        'patch_range': patch_range,
    }
    return patch_norm, meta


def compute_tile_starts(length, tile_size, stride):
    """Compute standard-size tile starts, back-projecting the last tile from the edge."""
    if tile_size <= 0:
        raise ValueError(f'tile_size must be positive, got {tile_size}.')
    if stride <= 0:
        raise ValueError(f'stride must be positive, got {stride}.')
    if stride > tile_size:
        raise ValueError(f'stride must be <= tile_size to avoid uncovered gaps, got {stride} > {tile_size}.')

    if length <= tile_size:
        return [0]

    last_start = length - tile_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _axis_weight(tile_size, starts, index, cut):
    """Create a 1D linear blend weight for one tiled axis."""
    weight = np.ones(tile_size, dtype=np.float32)
    start = starts[index]

    if index > 0:
        prev_overlap = starts[index - 1] + tile_size - start
        if prev_overlap > 0:
            weight[:prev_overlap] *= np.linspace(0, 1, prev_overlap, dtype=np.float32)
            if cut > 0 and prev_overlap > 2 * cut:
                weight[:cut] = 0

    if index < len(starts) - 1:
        next_overlap = start + tile_size - starts[index + 1]
        if next_overlap > 0:
            weight[-next_overlap:] *= np.linspace(1, 0, next_overlap, dtype=np.float32)
            if cut > 0 and next_overlap > 2 * cut:
                weight[-cut:] = 0

    return weight


def make_weight_3d(tile_size, z_starts, y_starts, x_starts, z_idx, y_idx, x_idx, cut=0):
    """Create a 3D separable weight map using the true overlap of neighboring tiles."""
    tile_z, tile_y, tile_x = tile_size
    wz = _axis_weight(tile_z, z_starts, z_idx, cut)
    wy = _axis_weight(tile_y, y_starts, y_idx, cut)
    wx = _axis_weight(tile_x, x_starts, x_idx, cut)
    return wz[:, None, None] * wy[None, :, None] * wx[None, None, :]


def _tensor_output_to_volume(output_res):
    """Convert a single-network output tensor to a (D, H, W) numpy volume."""
    output_np = output_res.detach().cpu().float().numpy()
    if output_np.ndim == 5:
        return output_np[0, 0]
    if output_np.ndim == 4:
        return output_np[0]
    if output_np.ndim == 3:
        return output_np
    return np.squeeze(output_np)


def process_whole_stack_40xsr(
        img_norm,
        model,
        device,
        tile_opt,
        boundary_pixel=0,
        cut_pixel=1,
        patch_min_max=True,
        logger=None):
    """Run 40xSR tiled inference with patch-level min-max normalization and weighted blending."""
    if logger is None:
        logger = logging.getLogger('basicsr')

    if img_norm.ndim != 3:
        raise ValueError(f'Input stack must be 3D (D, H, W), got {img_norm.shape}.')

    original_shape = img_norm.shape
    work_volume = img_norm.astype(np.float32, copy=False)

    if boundary_pixel > 0:
        pad_width = ((boundary_pixel, boundary_pixel),) * 3
        work_volume = np.pad(work_volume, pad_width, mode='constant', constant_values=0)

    tile_size = (
        int(_tile_attr(tile_opt, 'img_s')),
        int(_tile_attr(tile_opt, 'img_h')),
        int(_tile_attr(tile_opt, 'img_w')),
    )
    stride = (
        int(_tile_attr(tile_opt, 'gap_s')),
        int(_tile_attr(tile_opt, 'gap_h')),
        int(_tile_attr(tile_opt, 'gap_w')),
    )

    pad_after = tuple(max(tile - size, 0) for tile, size in zip(tile_size, work_volume.shape))
    if any(pad_after):
        logger.info('Input volume smaller than tile size in at least one axis; padding after by %s.', pad_after)
        work_volume = np.pad(
            work_volume,
            tuple((0, value) for value in pad_after),
            mode='constant',
            constant_values=0)

    work_shape = work_volume.shape
    z_starts = compute_tile_starts(work_shape[0], tile_size[0], stride[0])
    y_starts = compute_tile_starts(work_shape[1], tile_size[1], stride[1])
    x_starts = compute_tile_starts(work_shape[2], tile_size[2], stride[2])

    canvas_sum = np.zeros(work_shape, dtype=np.float32)
    canvas_weight = np.zeros(work_shape, dtype=np.float32)
    total_tiles = len(z_starts) * len(y_starts) * len(x_starts)
    tile_count = 0

    for z_idx, z_start in enumerate(z_starts):
        for y_idx, y_start in enumerate(y_starts):
            for x_idx, x_start in enumerate(x_starts):
                patch = work_volume[
                    z_start:z_start + tile_size[0],
                    y_start:y_start + tile_size[1],
                    x_start:x_start + tile_size[2]]

                if patch_min_max:
                    patch_input, meta = normalize_patch_min_max(patch)
                else:
                    patch_input = patch.astype(np.float32, copy=False)
                    meta = {
                        'patch_min': float(patch_input.min()),
                        'patch_max': float(patch_input.max()),
                        'patch_range': 1.0,
                    }

                batch_meta = {
                    'is_front': z_idx == 0,
                    'is_back': z_idx == len(z_starts) - 1,
                    'is_top': y_idx == 0,
                    'is_bottom': y_idx == len(y_starts) - 1,
                    'is_left': x_idx == 0,
                    'is_right': x_idx == len(x_starts) - 1,
                    'z_start': z_start,
                    'y_start': y_start,
                    'x_start': x_start,
                    'actual_s': tile_size[0],
                    'actual_h': tile_size[1],
                    'actual_w': tile_size[2],
                    'patch_size': tile_size,
                    'cut': cut_pixel,
                    **meta,
                }

                input_data = torch.from_numpy(np.ascontiguousarray(patch_input)).unsqueeze(dim=0).unsqueeze(dim=0)
                input_data = input_data.to(device)

                model_out = model(input_data)
                if isinstance(model_out, (list, tuple)):
                    output_res = model_out[-1]
                else:
                    output_res = model_out

                output_vol = _tensor_output_to_volume(output_res)
                if output_vol.shape != tile_size:
                    raise ValueError(
                        f'Model output shape {output_vol.shape} does not match tile size {tile_size} '
                        f'at z/y/x start ({z_start}, {y_start}, {x_start}).')

                out_01 = np.clip(output_vol.astype(np.float32, copy=False), 0, 1)
                if patch_min_max:
                    if batch_meta['patch_range'] > 1e-10:
                        out_restored = out_01 * batch_meta['patch_range']
                    else:
                        out_restored = np.zeros_like(out_01, dtype=np.float32)
                else:
                    out_restored = out_01

                weight = make_weight_3d(
                    tile_size, z_starts, y_starts, x_starts, z_idx, y_idx, x_idx, cut=cut_pixel)

                z_end = z_start + tile_size[0]
                y_end = y_start + tile_size[1]
                x_end = x_start + tile_size[2]
                canvas_sum[z_start:z_end, y_start:y_end, x_start:x_end] += out_restored * weight
                canvas_weight[z_start:z_end, y_start:y_end, x_start:x_end] += weight

                del model_out, input_data, output_res
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                tile_count += 1
                logger.info(
                    'Processed tile %d/%d at z/y/x start (%d, %d, %d), patch range %.6g.',
                    tile_count, total_tiles, z_start, y_start, x_start, batch_meta['patch_range'])

    if np.any(canvas_weight <= 0):
        zero_count = int(np.count_nonzero(canvas_weight <= 0))
        raise RuntimeError(f'Tiled blending produced {zero_count} zero-weight voxels.')

    output = (canvas_sum / canvas_weight).astype(np.float32, copy=False)
    output[output < 0] = 0

    if any(pad_after):
        crop_shape = tuple(size - pad for size, pad in zip(work_shape, pad_after))
        output = output[
            :crop_shape[0],
            :crop_shape[1],
            :crop_shape[2]]

    if boundary_pixel > 0:
        output = output[
            boundary_pixel:boundary_pixel + original_shape[0],
            boundary_pixel:boundary_pixel + original_shape[1],
            boundary_pixel:boundary_pixel + original_shape[2]]

    if output.shape != original_shape:
        raise RuntimeError(f'Output shape {output.shape} does not match input shape {original_shape}.')

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tiled 3D inference for large microscopy TIFF stacks (BasicSR).')
    parser.add_argument('--opt', type=str, default=DEFAULT_OPT, help='Path to train/test YAML with network_g.')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to net_g_*.pth checkpoint.')
    parser.add_argument('--data_folder', type=str, default=DEFAULT_DATA_FOLDER, help='Flat directory of input TIFFs.')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory for predicted TIFFs.')
    parser.add_argument('--param_key', type=str, default=DEFAULT_PARAM_KEY, choices=['params_ema', 'params'])
    parser.add_argument('--gpu', type=int, default=DEFAULT_GPU, help='GPU id; use -1 for CPU.')
    parser.add_argument('--strict_load', action='store_true', default=DEFAULT_STRICT_LOAD)
    parser.add_argument('--no_strict_load', dest='strict_load', action='store_false')
    parser.add_argument(
        '--auto_latest',
        action='store_true',
        help='Auto-pick latest net_g_*.pth from experiments/{name}/models/.')
    parser.add_argument('--img_w', type=int, default=DEFAULT_IMG_W, help='Patch width.')
    parser.add_argument('--img_h', type=int, default=DEFAULT_IMG_H, help='Patch height.')
    parser.add_argument('--img_s', type=int, default=DEFAULT_IMG_S, help='Patch depth (slices).')
    parser.add_argument('--gap_w', type=int, default=DEFAULT_GAP_W, help='Stride along width.')
    parser.add_argument('--gap_h', type=int, default=DEFAULT_GAP_H, help='Stride along height.')
    parser.add_argument('--gap_s', type=int, default=DEFAULT_GAP_S, help='Stride along depth.')
    parser.add_argument('--boundary_pixel', type=int, default=DEFAULT_BOUNDARY_PIXEL,
                        help='Border crop after stitching.')
    parser.add_argument('--cut_pixel', type=int, default=DEFAULT_CUT_PIXEL,
                        help='Overlap trim for stitchingZ/stitchingXY.')
    parser.add_argument('--start_num', type=int, default=DEFAULT_START_NUM, help='First stack index to process.')
    parser.add_argument('--end_num', type=int, default=DEFAULT_END_NUM, help='Last stack index (exclusive).')
    parser.add_argument('--xy_resolution', type=float, default=DEFAULT_XY_RESOLUTION,
                        help='XY resolution (micron/pixel) for ImageJ metadata.')
    parser.add_argument('--z_spacing', type=float, default=DEFAULT_Z_SPACING,
                        help='Z spacing (micron) for ImageJ metadata.')
    parser.add_argument('--percentile_low', type=float, default=DEFAULT_PERCENTILE_LOW,
                        help='Lower percentile for whole-volume normalization.')
    parser.add_argument('--percentile_high', type=float, default=DEFAULT_PERCENTILE_HIGH,
                        help='Upper percentile for whole-volume normalization.')
    parser.add_argument('--patch_min_max', action='store_true', default=DEFAULT_PATCH_MIN_MAX,
                        help='Apply per-patch min-max normalization before network inference.')
    parser.add_argument('--no_patch_min_max', dest='patch_min_max', action='store_false',
                        help='Disable per-patch min-max normalization.')
    parser.add_argument(
        '--sort_by_time',
        action='store_true',
        help='Sort input TIFFs by T index in filename (e.g. stack_T001.tif). '
             'Default: natural alphanumeric order by filename.')
    return parser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger('basicsr')

    opt_path = resolve_path(args.opt)
    data_folder = resolve_path(args.data_folder)
    output_dir = resolve_path(args.output_dir)

    if not data_folder:
        raise ValueError('Set --data_folder or edit DEFAULT_DATA_FOLDER.')
    if not output_dir:
        raise ValueError('Set --output_dir or edit DEFAULT_OUTPUT_DIR.')

    yaml_opt = load_opt(opt_path)
    manual_seed = yaml_opt.get('manual_seed', 10)
    set_random_seed(manual_seed)

    if args.auto_latest:
        checkpoint_path = find_latest_checkpoint(opt_path)
    else:
        checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else None
        if not checkpoint_path or not osp.isfile(checkpoint_path):
            yaml_ckpt = yaml_opt.get('path', {}).get('pretrain_network_g')
            if yaml_ckpt:
                checkpoint_path = resolve_path(yaml_ckpt, base_dir=ROOT)
    if not checkpoint_path or not osp.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {checkpoint_path}\n'
            'Set --checkpoint, use --auto_latest, or set path.pretrain_network_g in the YAML.')

    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')

    net = build_network(yaml_opt['network_g'])
    load_network_weights(net, checkpoint_path, strict=args.strict_load, param_key=args.param_key)
    net = net.to(device)
    net.eval()

    tile_opt = build_tile_opt(args, yaml_opt)
    img_paths = scan_flat_tiff_paths(data_folder, sort_by_time=args.sort_by_time)
    os.makedirs(output_dir, exist_ok=True)

    start_num = args.start_num
    end_num = len(img_paths) if args.end_num >= len(img_paths) else args.end_num
    if start_num >= end_num:
        raise ValueError('end_num must be greater than start_num.')

    logger.info('Option file: %s', opt_path)
    logger.info('Checkpoint: %s', checkpoint_path)
    logger.info('Data folder: %s', data_folder)
    logger.info('Output dir: %s', output_dir)
    logger.info('Device: %s', device)
    logger.info('Tile size (s,h,w): (%d, %d, %d)', tile_opt.img_s, tile_opt.img_h, tile_opt.img_w)
    logger.info('Tile gap (s,h,w): (%d, %d, %d)', tile_opt.gap_s, tile_opt.gap_h, tile_opt.gap_w)
    logger.info(
        'Whole-volume percentile normalization: %.4g / %.4g; patch min-max: %s',
        args.percentile_low, args.percentile_high, args.patch_min_max)
    logger.info('Stacks to process: [%d, %d) of %d', start_num, end_num, len(img_paths))

    with torch.no_grad():
        for stack_idx in range(start_num, end_num):
            img_path = img_paths[stack_idx]
            img_name = osp.basename(img_path)
            stem = osp.splitext(img_name)[0]
            save_path = osp.join(output_dir, f'{stem}_DeconNet_output.tif')

            logger.info('[%d/%d] Processing %s', stack_idx - start_num + 1, end_num - start_num, img_name)

            img = imread(img_path).astype(np.float32)
            if img.ndim != 3:
                raise ValueError(f'Input stack must be 3D (D, H, W), got {img.shape}: {img_path}')

            img = normalize_percentile(
                img,
                percentile_low=args.percentile_low,
                percentile_high=args.percentile_high)

            output = process_whole_stack_40xsr(
                img,
                net,
                device,
                tile_opt,
                boundary_pixel=args.boundary_pixel,
                cut_pixel=args.cut_pixel,
                patch_min_max=args.patch_min_max,
                logger=logger,
            )

            imwrite(
                save_path,
                output.astype(np.float32, copy=False),
                imagej=True,
                bigtiff=False,
                resolution=(1. / args.xy_resolution, 1. / args.xy_resolution),
                metadata={
                    'spacing': args.z_spacing,
                    'unit': 'micron',
                    'axes': 'ZYX',
                },
            )
            logger.info('Saved %s', save_path)

    logger.info('Tiled inference finished. %d volumes saved to %s', end_num - start_num, output_dir)


if __name__ == '__main__':
    main(parse_args())
