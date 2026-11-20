"""Standalone 3D-FUDIPn inference script for microscopy TIFF stacks.

Supports IDE one-click run (edit DEFAULT_* below) and CLI overrides via argparse.
"""

import argparse
import glob
import logging
import os
import sys
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
from tifffile import imread

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
from basicsr.utils import imwrite_volume, set_random_seed, tensor2volume, yaml_load


# --- LLSM datast - ER metaphase - SNR 5 ---
DEFAULT_OPT = '/path_to_config_file/train_FUDIPn2stage_config.yml'
DEFAULT_CHECKPOINT = '/path_to_ckpt/net_g_25000.pth'
DEFAULT_INPUT_DIR = '/paired_dataset/test'
DEFAULT_OUTPUT_DIR = '/paired_dataset/test/FUDIPn_output'
DEFAULT_LQ_SUBDIR = 'input'
DEFAULT_PARAM_KEY = 'params_ema'
DEFAULT_GPU = 0
DEFAULT_STRICT_LOAD = True
# -------------------------------------------------

def resolve_path(path, base_dir=ROOT):
    """Resolve a path relative to repo root when not absolute."""
    if path is None:
        return None
    path = osp.expanduser(path)
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(base_dir, path))


def scan_lq_paths(input_dir, lq_subdir):
    """Collect sorted 3D TIFF paths from input_dir/lq_subdir."""
    lq_dir = osp.join(input_dir, lq_subdir)
    if not osp.isdir(lq_dir):
        raise FileNotFoundError(f'LQ directory not found: {lq_dir}')

    suffixes = ('*.tif', '*.tiff', '*.TIF', '*.TIFF')
    paths = []
    for suffix in suffixes:
        paths.extend(glob.glob(osp.join(lq_dir, suffix)))
    paths = sorted(set(paths))
    if len(paths) == 0:
        raise FileNotFoundError(f'No TIFF stack files found in: {lq_dir}')
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


def parse_args():
    parser = argparse.ArgumentParser(description='3D RCAN inference for microscopy TIFF stacks.')
    parser.add_argument('--opt', type=str, default=DEFAULT_OPT, help='Path to train/test YAML with network_g.')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to net_g_*.pth checkpoint.')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR, help='Dataset root directory.')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory for predicted TIFFs.')
    parser.add_argument('--lq_subdir', type=str, default=DEFAULT_LQ_SUBDIR, help='LQ subfolder under input_dir.')
    parser.add_argument('--param_key', type=str, default=DEFAULT_PARAM_KEY, choices=['params_ema', 'params'])
    parser.add_argument('--gpu', type=int, default=DEFAULT_GPU, help='GPU id; use -1 for CPU.')
    parser.add_argument('--strict_load', action='store_true', default=DEFAULT_STRICT_LOAD)
    parser.add_argument('--no_strict_load', dest='strict_load', action='store_false')
    parser.add_argument(
        '--auto_latest',
        action='store_true',
        help='Auto-pick latest net_g_*.pth from experiments/{name}/models/.')
    return parser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger('basicsr')

    opt_path = resolve_path(args.opt)
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    opt = load_opt(opt_path)
    manual_seed = opt.get('manual_seed', 10)
    set_random_seed(manual_seed)

    if args.auto_latest:
        checkpoint_path = find_latest_checkpoint(opt_path)
    else:
        checkpoint_path = resolve_path(args.checkpoint)
        if not osp.isfile(checkpoint_path):
            yaml_ckpt = opt.get('path', {}).get('pretrain_network_g')
            if yaml_ckpt:
                checkpoint_path = resolve_path(yaml_ckpt, base_dir=ROOT)
    if not osp.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {checkpoint_path}\n'
            'Set --checkpoint, use --auto_latest, or set path.pretrain_network_g in the YAML.')

    if args.gpu < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')

    net = build_network(opt['network_g'])
    load_network_weights(net, checkpoint_path, strict=args.strict_load, param_key=args.param_key)
    net = net.to(device)
    net.eval()

    lq_paths = scan_lq_paths(input_dir, args.lq_subdir)
    os.makedirs(output_dir, exist_ok=True)

    logger.info('Option file: %s', opt_path)
    logger.info('Checkpoint: %s', checkpoint_path)
    logger.info('Input LQ dir: %s', osp.join(input_dir, args.lq_subdir))
    logger.info('Output dir: %s', output_dir)
    logger.info('Device: %s', device)
    logger.info('Volumes to process: %d', len(lq_paths))

    with torch.no_grad():
        for idx, lq_path in enumerate(lq_paths):
            img_name = osp.basename(lq_path)
            save_path = osp.join(output_dir, img_name)

            lq_vol = imread(lq_path).astype(np.float32, copy=False)
            if lq_vol.ndim != 3:
                raise ValueError(f'LQ stack must be 3D (D, H, W), got {lq_vol.shape}: {lq_path}')

            lq_tensor = torch.from_numpy(np.ascontiguousarray(lq_vol)).unsqueeze(0).unsqueeze(0).to(device)
            output = net(lq_tensor)
            if isinstance(output, (list, tuple)):
                output = output[-1]

            volume = tensor2volume(output)[0]
            imwrite_volume(volume, save_path)
            logger.info('[%d/%d] Saved %s', idx + 1, len(lq_paths), save_path)

    logger.info('Inference finished. %d volumes saved to %s', len(lq_paths), output_dir)


if __name__ == '__main__':
    main(parse_args())
