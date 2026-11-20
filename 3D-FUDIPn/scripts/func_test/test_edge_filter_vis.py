"""
Visualize 3D gradient magnitude maps produced by GradEdge3D.

PyCharm usage:
    Edit USER_CONFIG below, then click Run. No CLI arguments required.

Optional CLI overrides:
    python scripts/test_edge_filter_vis.py \\
        --input-path path/to/volume.tif --output-dir path/to/output
"""

import argparse
import os
import sys
from os import path as osp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile import imread

# Ensure repo root is importable when running from PyCharm or scripts/.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.losses.edge_loss import GradEdge3D
from basicsr.utils.volume_util import imwrite_volume, volume2preview

# --------------- PyCharm one-click config ---------------
USER_CONFIG = {
    # 'input_path': r'E:/dataset/3D-RCAN-Denoising1_dataset/Denoising_paired_patches/Lysosome/test/gt/000004_Lysosome_1.tif',
    # 'input_path': r'D:/Code/FUDIP_proj/fudipn-basicsr/experiments/train_FUDIPn_s2d4_denosing_task_lyso/visualization/000004_Lysosome_1/000004_Lysosome_1_19000.tif',
    # 'input_path': r'E:/dataset/3D-RCAN-Denoising1_dataset/Denoising_paired_patches/Actin/test/gt/000007_Actin_17.tif',
    # 'output_dir': r'D:/Code/FUDIP_proj/fudipn-basicsr/experiments/vis_edge_loss/Actin_gt_007/',

    # 'input_path': r'E:/dataset/3D-RCAN-Denoising1_dataset/Denoising_paired_patches/Tomm20_Mitochondria/train/gt/000001_Tomm20_Mitochondria_image000005.tif',
    # 'output_dir': r'D:/Code/FUDIP_proj/fudipn-basicsr/experiments/vis_edge_loss/Mito_gt_001/',

    'input_path': r'E:/dataset/BioTISR_dataset/LLSM_dataset/paired_dataset_128/Inner_Mito_Membrane/SNR_3/test/gt/00000001.tif',
    'output_dir': r'D:/Code/FUDIP_proj/fudipn-basicsr/experiments/vis_edge_loss/LLSM_Mito_gt_001/',

    'n1': 1,
    'n2': 2,
    'n3': 2,
    'normalize_edge': False,
    'device': 'cuda',  # 'cuda', 'cpu', or 'auto'
    'grid_size': 4,
    'dpi': 150,
}
# --------------------------------------------------------


def resolve_device(device_name):
    """Resolve execution device from config string."""
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_name)


def load_volume(input_path):
    """Load a single 3D TIFF volume with shape (D, H, W)."""
    if not osp.isfile(input_path):
        raise FileNotFoundError(f'Input TIFF not found: {input_path}')

    volume = imread(input_path)
    if volume.ndim != 3:
        raise ValueError(
            f'Input stack must have shape (D, H, W), but got {volume.shape}: {input_path}')

    volume = np.ascontiguousarray(volume.astype(np.float32, copy=False))
    return volume


def compute_gradient(volume_np, n1, n2, n3, normalize_edge, device):
    """Run GradEdge3D on a single-channel 3D volume."""
    tensor = torch.from_numpy(volume_np).unsqueeze(0).unsqueeze(0).to(device)
    edge_filter = GradEdge3D(n1, n2, n3, normalize=normalize_edge).to(device).eval()

    with torch.no_grad():
        grad = edge_filter(tensor)

    grad_np = grad.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
    return grad, grad_np


def sample_z_indices(depth, grid_size):
    """Uniformly sample z indices for a grid_size x grid_size panel."""
    num_panels = grid_size * grid_size
    if depth <= 0:
        raise ValueError(f'Volume depth must be positive, but got D={depth}.')
    if depth >= num_panels:
        return np.linspace(0, depth - 1, num_panels, dtype=int)
    return np.arange(depth, dtype=int)


def save_z_grid_figure(grad_np, output_path, grid_size=4, dpi=150):
    """Save a grid_size x grid_size figure of gradient maps at sampled z slices."""
    depth = grad_np.shape[0]
    z_indices = sample_z_indices(depth, grid_size)
    sampled = grad_np[z_indices]

    vmin = np.percentile(sampled, 2)
    vmax = np.percentile(sampled, 98)
    if vmax <= vmin:
        vmin = float(grad_np.min())
        vmax = float(grad_np.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(2.5 * grid_size, 2.5 * grid_size))
    axes = np.atleast_2d(axes)

    for panel_idx, ax in enumerate(axes.flat):
        if panel_idx < len(z_indices):
            z_idx = int(z_indices[panel_idx])
            ax.imshow(grad_np[z_idx], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f'z={z_idx}', fontsize=9)
        else:
            ax.axis('off')
            ax.set_visible(False)
            continue
        ax.axis('off')

    fig.suptitle('Gradient magnitude (sampled z slices)', fontsize=12)
    fig.tight_layout()

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_mip_figure(grad_tensor, grad_np, output_path, dpi=150):
    """Save maximum-intensity projection of the gradient volume."""
    mip = volume2preview(grad_tensor, mode='mip').squeeze().numpy()

    vmin = np.percentile(grad_np, 2)
    vmax = np.percentile(grad_np, 98)
    if vmax <= vmin:
        vmin = float(grad_np.min())
        vmax = float(grad_np.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mip, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title('Gradient magnitude MIP (z-axis)')
    ax.axis('off')
    fig.tight_layout()

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def run_edge_filter_vis(cfg):
    """Load volume, compute gradient map, save TIFF and preview figures."""
    device = resolve_device(cfg['device'])
    volume_np = load_volume(cfg['input_path'])
    grad_tensor, grad_np = compute_gradient(
        volume_np,
        cfg['n1'],
        cfg['n2'],
        cfg['n3'],
        cfg['normalize_edge'],
        device,
    )

    os.makedirs(cfg['output_dir'], exist_ok=True)

    tiff_path = osp.join(cfg['output_dir'], 'gradient_mag.tif')
    grid_path = osp.join(cfg['output_dir'], 'gradient_z_grid.png')
    mip_path = osp.join(cfg['output_dir'], 'gradient_mip.png')

    imwrite_volume(grad_np, tiff_path)
    save_z_grid_figure(grad_np, grid_path, grid_size=cfg['grid_size'], dpi=cfg['dpi'])
    save_mip_figure(grad_tensor, grad_np, mip_path, dpi=cfg['dpi'])

    print('=' * 60)
    print('GradEdge3D visualization complete')
    print(f'Input path      : {cfg["input_path"]}')
    print(f'Input shape     : {tuple(volume_np.shape)} (D, H, W)')
    print(f'Gradient shape  : {tuple(grad_np.shape)} (D, H, W)')
    print(f'Gradient min    : {grad_np.min():.6f}')
    print(f'Gradient max    : {grad_np.max():.6f}')
    print(f'Gradient mean   : {grad_np.mean():.6f}')
    print(f'Device          : {device}')
    print(f'normalize_edge  : {cfg["normalize_edge"]}')
    print(f'Gradient TIFF   : {tiff_path}')
    print(f'Z-grid figure   : {grid_path}')
    print(f'MIP figure      : {mip_path}')
    print('=' * 60)


def build_parser(defaults):
    """Build argparse parser with USER_CONFIG defaults."""
    parser = argparse.ArgumentParser(description='Visualize GradEdge3D gradient maps.')
    parser.add_argument(
        '--input-path',
        default=defaults['input_path'],
        help='Path to input 3D TIFF stack with shape (D, H, W).',
    )
    parser.add_argument(
        '--output-dir',
        default=defaults['output_dir'],
        help='Directory for gradient TIFF and preview figures.',
    )
    parser.add_argument('--n1', type=int, default=defaults['n1'], help='Sobel kernel parameter n1.')
    parser.add_argument('--n2', type=int, default=defaults['n2'], help='Sobel kernel parameter n2.')
    parser.add_argument('--n3', type=int, default=defaults['n3'], help='Sobel kernel parameter n3.')
    parser.add_argument(
        '--normalize-edge',
        action='store_true',
        default=defaults['normalize_edge'],
        help='Apply instance-wise min-max normalization in GradEdge3D.',
    )
    parser.add_argument(
        '--device',
        default=defaults['device'],
        choices=['auto', 'cuda', 'cpu'],
        help='Execution device.',
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=defaults['grid_size'],
        help='Grid dimension for z-slice visualization (grid_size x grid_size panels).',
    )
    parser.add_argument('--dpi', type=int, default=defaults['dpi'], help='DPI for saved PNG figures.')
    return parser


def args_to_config(args):
    """Convert argparse namespace to runtime config dict."""
    return {
        'input_path': args.input_path,
        'output_dir': args.output_dir,
        'n1': args.n1,
        'n2': args.n2,
        'n3': args.n3,
        'normalize_edge': args.normalize_edge,
        'device': args.device,
        'grid_size': args.grid_size,
        'dpi': args.dpi,
    }


def main():
    parser = build_parser(USER_CONFIG)
    args = parser.parse_args()
    cfg = args_to_config(args)
    run_edge_filter_vis(cfg)


if __name__ == '__main__':
    main()
