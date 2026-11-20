import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite


TIFF_SUFFIXES = {'.tif', '.tiff'}
RAW_DIR_NAMES = ('Raw', 'RAW', 'raw', 'Low', 'LOW', 'low', 'LQ', 'lq')
GT_DIR_NAMES = ('GT', 'gt')
TEST_GT_DIR_NAMES = ('Test_GT', 'TEST_GT', 'test_gt', 'GT', 'gt')
# Edit this block when running the script directly from PyCharm without command-line arguments.
USE_IDE_CONFIG_WHEN_NO_CLI_ARGS = True
IDE_CONFIG = {
    'input_root': Path('E:/dataset/3D-RCAN-Denoising1_dataset/Denoising'),
    'output_root': Path('E:/dataset/3D-RCAN-Denoising1_dataset/Denoising_paired_patches'),
    # Use 'train', 'test', 'all', 'both', or ['train', 'test'].
    'split': 'train',
    'categories': ['Actin'],
    'num_patches': 10,
    'patch_size': (16, 192, 192),
    'discard_z': 2,
    'discard_xy': 10,
    'percentile_low': 0.01,
    'percentile_high': 99.99,
    'intensity_threshold': 0.11,
    'area_ratio_threshold': 0.008,
    'raw_background_threshold': 0.15,
    'max_attempts': None,
    'seed': 0,
    'overwrite': False,
}


@dataclass(frozen=True)
class StackPair:
    category: str
    key: str
    raw_path: Path
    gt_path: Path


def normalize_percentile(im, low=0.2, high=99.9):
    """Normalize an array using percentile-based min-max scaling."""
    p_low, p_high = np.percentile(im, low), np.percentile(im, high)
    return normalize_min_max(im, max_v=p_high, min_v=p_low)


def normalize_min_max(im, max_v, min_v=0):
    """Normalize an array with explicit lower and upper intensity bounds."""
    eps = 1e-10
    im = (im - min_v) / (max_v - min_v + eps)
    return im.astype(np.float32, copy=False)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('value must be a non-negative integer')
    return value


def validate_percentiles(low, high):
    if not (0 <= low < high <= 100):
        raise ValueError(f'Percentiles must satisfy 0 <= low < high <= 100, but got {low} and {high}.')

def normalize_splits(split_value):
    """Normalize split configuration into an ordered list of dataset splits."""
    if isinstance(split_value, str):
        split_items = [split_value]
    else:
        split_items = list(split_value)

    normalized_splits = []
    for item in split_items:
        split = str(item).lower()
        if split in ('all', 'both'):
            normalized_splits.extend(['train', 'test'])
        elif split in ('train', 'test'):
            normalized_splits.append(split)
        else:
            raise ValueError(f'Split must be train, test, all, or both, but got {item}.')

    ordered_unique_splits = []
    for split in ('train', 'test'):
        if split in normalized_splits:
            ordered_unique_splits.append(split)

    if not ordered_unique_splits:
        raise ValueError('At least one split must be specified.')
    return ordered_unique_splits

def args_from_ide_config(config):
    """Build argparse-compatible arguments from the editable IDE configuration block."""
    required_keys = (
        'input_root',
        'output_root',
        'split',
        'categories',
        'num_patches',
        'patch_size',
        'discard_z',
        'discard_xy',
        'percentile_low',
        'percentile_high',
        'intensity_threshold',
        'area_ratio_threshold',
        'raw_background_threshold',
        'max_attempts',
        'seed',
        'overwrite',
    )
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f'IDE_CONFIG is missing required keys: {missing_keys}')

    normalized = dict(config)
    normalized['input_root'] = Path(normalized['input_root'])
    normalized['output_root'] = Path(normalized['output_root'])

    categories = normalized['categories']
    if isinstance(categories, str):
        normalized['categories'] = [categories]
    else:
        normalized['categories'] = list(categories)

    normalized['split'] = normalize_splits(normalized['split'])
    normalized['patch_size'] = tuple(int(value) for value in normalized['patch_size'])
    normalized['num_patches'] = int(normalized['num_patches'])
    normalized['discard_z'] = int(normalized['discard_z'])
    normalized['discard_xy'] = int(normalized['discard_xy'])
    normalized['percentile_low'] = float(normalized['percentile_low'])
    normalized['percentile_high'] = float(normalized['percentile_high'])
    normalized['intensity_threshold'] = float(normalized['intensity_threshold'])
    normalized['area_ratio_threshold'] = float(normalized['area_ratio_threshold'])
    normalized['raw_background_threshold'] = float(normalized['raw_background_threshold'])
    normalized['seed'] = int(normalized['seed'])
    normalized['overwrite'] = bool(normalized['overwrite'])
    if normalized['max_attempts'] is not None:
        normalized['max_attempts'] = int(normalized['max_attempts'])

    if len(normalized['patch_size']) != 3:
        raise ValueError(f'IDE_CONFIG patch_size must contain exactly 3 values, but got {normalized["patch_size"]}.')
    if normalized['num_patches'] <= 0:
        raise ValueError(f'IDE_CONFIG num_patches must be positive, but got {normalized["num_patches"]}.')
    if any(value <= 0 for value in normalized['patch_size']):
        raise ValueError(f'IDE_CONFIG patch_size values must be positive, but got {normalized["patch_size"]}.')
    if normalized['discard_z'] < 0 or normalized['discard_xy'] < 0:
        raise ValueError(
            f'IDE_CONFIG discard_z and discard_xy must be non-negative, but got '
            f'{normalized["discard_z"]} and {normalized["discard_xy"]}.')
    if normalized['max_attempts'] is not None and normalized['max_attempts'] <= 0:
        raise ValueError(f'IDE_CONFIG max_attempts must be positive or None, but got {normalized["max_attempts"]}.')

    return argparse.Namespace(**normalized)


def is_tiff_file(path):
    return path.is_file() and path.suffix.lower() in TIFF_SUFFIXES


def scan_tiff_files(folder):
    paths = sorted(path for path in folder.iterdir() if is_tiff_file(path))
    if not paths:
        raise FileNotFoundError(f'No TIFF files found in folder: {folder}')
    return paths


def find_child_dir(parent, names, required=True):
    children = {path.name.lower(): path for path in parent.iterdir() if path.is_dir()}
    for name in names:
        path = children.get(name.lower())
        if path is not None:
            return path

    if required:
        raise FileNotFoundError(f'Cannot find any of {names} under: {parent}')
    return None


def parse_categories(values):
    categories = []
    for value in values:
        categories.extend(item.strip() for item in value.split(','))
    categories = [item for item in categories if item]
    if not categories:
        raise ValueError('At least one category must be specified.')
    return categories


def resolve_categories(input_root, requested_categories):
    if any(category.lower() == 'all' for category in requested_categories):
        categories = sorted(path.name for path in input_root.iterdir() if path.is_dir())
        if not categories:
            raise FileNotFoundError(f'No category folders found under: {input_root}')
        return categories

    resolved = []
    available = {path.name.lower(): path.name for path in input_root.iterdir() if path.is_dir()}
    for category in requested_categories:
        matched_name = available.get(category.lower())
        if matched_name is None:
            raise FileNotFoundError(f'Category "{category}" is not found under: {input_root}')
        resolved.append(matched_name)
    return resolved


def normalize_pair_key(path):
    """Create a filename key for pairing raw and GT TIFF stacks."""
    key = path.stem.lower()
    key = re.sub(r'_decon$', '', key)
    key = re.sub(r'^\d+(?:\.\d+)?%', '', key)
    key = re.sub(r'^c\d+-', '', key)
    key = re.sub(r'^(?:high|low)[\s_-]*', '', key)
    key = re.sub(r'(?:raw|gt)$', '', key)
    key = re.sub(r'[\s_-]+$', '', key)
    return key


def build_key_map(paths, role, category, split):
    key_map = {}
    for path in paths:
        key = normalize_pair_key(path)
        if key in key_map:
            raise ValueError(
                f'Duplicate {role} pairing key "{key}" in category "{category}" split "{split}": '
                f'{key_map[key]} and {path}')
        key_map[key] = path
    return key_map


def format_missing_keys(keys, key_map):
    names = [key_map[key].name for key in sorted(keys)]
    preview = ', '.join(names[:10])
    if len(names) > 10:
        preview += f', ... ({len(names)} files total)'
    return preview


def pair_stack_paths(raw_paths, gt_paths, category, split):
    raw_map = build_key_map(raw_paths, 'Raw', category, split)
    gt_map = build_key_map(gt_paths, 'GT', category, split)

    raw_keys = set(raw_map)
    gt_keys = set(gt_map)
    missing_gt = raw_keys - gt_keys
    missing_raw = gt_keys - raw_keys
    if missing_gt or missing_raw:
        details = []
        if missing_gt:
            details.append(f'Raw files without GT: {format_missing_keys(missing_gt, raw_map)}')
        if missing_raw:
            details.append(f'GT files without Raw: {format_missing_keys(missing_raw, gt_map)}')
        raise ValueError(
            f'Unpaired TIFF stacks found in category "{category}" split "{split}". ' + ' '.join(details))

    return [
        StackPair(category=category, key=key, raw_path=raw_map[key], gt_path=gt_map[key])
        for key in sorted(raw_keys)
    ]


def get_split_dir(category_dir, split):
    if split == 'train':
        return find_child_dir(category_dir, ('Training', 'Train', 'training', 'train'))
    if split == 'test':
        return find_child_dir(category_dir, ('Test', 'test', 'TEST'))
    raise ValueError(f'Unsupported split: {split}')


def collect_category_pairs(input_root, category, split):
    category_dir = find_child_dir(input_root, (category,))
    split_dir = get_split_dir(category_dir, split)

    if split == 'train':
        raw_dir = find_child_dir(split_dir, RAW_DIR_NAMES)
        gt_dir = find_child_dir(split_dir, GT_DIR_NAMES)
        raw_paths = scan_tiff_files(raw_dir)
        gt_paths = scan_tiff_files(gt_dir)
    else:
        raw_dir = find_child_dir(split_dir, RAW_DIR_NAMES, required=False)
        gt_dir = find_child_dir(split_dir, TEST_GT_DIR_NAMES)
        raw_paths = scan_tiff_files(raw_dir) if raw_dir is not None else scan_tiff_files(split_dir)
        gt_paths = scan_tiff_files(gt_dir)

    return pair_stack_paths(raw_paths, gt_paths, category, split)


def collect_all_pairs(input_root, categories, split):
    pairs_by_category = {}
    for category in categories:
        pairs = collect_category_pairs(input_root, category, split)
        if not pairs:
            raise FileNotFoundError(f'No paired TIFF stacks found for category "{category}" split "{split}".')
        pairs_by_category[category] = pairs
    return pairs_by_category


def distribute_counts(total_count, items):
    base = total_count // len(items)
    remainder = total_count % len(items)
    return {item: base + (index < remainder) for index, item in enumerate(items)}


def build_pair_quotas(num_patches, pairs_by_category):
    category_names = sorted(pairs_by_category)
    category_quotas = distribute_counts(num_patches, category_names)
    pair_quotas = []
    for category in category_names:
        category_count = category_quotas[category]
        pairs = pairs_by_category[category]
        counts = distribute_counts(category_count, pairs)
        pair_quotas.extend((pair, count) for pair, count in counts.items() if count > 0)
    return pair_quotas


def crop_edges(volume, discard_z, discard_xy, label, path):
    z_size, y_size, x_size = volume.shape
    if discard_z * 2 >= z_size:
        raise ValueError(
            f'{label} z discard ({discard_z} from each side) removes all z slices from shape {volume.shape}: {path}')
    if discard_xy * 2 >= y_size or discard_xy * 2 >= x_size:
        raise ValueError(
            f'{label} xy discard ({discard_xy} from each side) removes all xy pixels from shape {volume.shape}: {path}')

    z_slice = slice(discard_z, z_size - discard_z if discard_z > 0 else None)
    y_slice = slice(discard_xy, y_size - discard_xy if discard_xy > 0 else None)
    x_slice = slice(discard_xy, x_size - discard_xy if discard_xy > 0 else None)
    return volume[z_slice, y_slice, x_slice]


def validate_volume_shape(volume, label, path):
    if volume.ndim != 3:
        raise ValueError(f'{label} stack must be a 3D TIFF with shape (Z, Y, X), but got {volume.shape}: {path}')


def validate_patch_size(volume_shape, patch_size, pair):
    z_size, y_size, x_size = volume_shape
    patch_z, patch_y, patch_x = patch_size
    if patch_z > z_size:
        raise ValueError(
            f'Patch z size {patch_z} is larger than cropped z size {z_size} for '
            f'{pair.category}/{pair.key}: {pair.raw_path} and {pair.gt_path}')
    if patch_y > y_size or patch_x > x_size:
        raise ValueError(
            f'Patch xy size {(patch_y, patch_x)} is larger than cropped xy size {(y_size, x_size)} for '
            f'{pair.category}/{pair.key}: {pair.raw_path} and {pair.gt_path}')


def load_and_prepare_pair(pair, patch_size, discard_z, discard_xy, percentile_low, percentile_high):
    raw = imread(pair.raw_path)
    gt = imread(pair.gt_path)

    validate_volume_shape(raw, 'Raw', pair.raw_path)
    validate_volume_shape(gt, 'GT', pair.gt_path)
    if raw.shape != gt.shape:
        raise ValueError(
            f'Raw and GT stack shapes differ for {pair.category}/{pair.key}: '
            f'{raw.shape} vs {gt.shape}. Raw: {pair.raw_path}. GT: {pair.gt_path}')

    raw = crop_edges(raw.astype(np.float32, copy=False), discard_z, discard_xy, 'Raw', pair.raw_path)
    gt = crop_edges(gt.astype(np.float32, copy=False), discard_z, discard_xy, 'GT', pair.gt_path)
    validate_patch_size(raw.shape, patch_size, pair)

    raw = normalize_percentile(raw, percentile_low, percentile_high)
    gt = normalize_percentile(gt, percentile_low, percentile_high)
    return raw, gt


def generate_3d_patch_v2(
        raw,
        gt,
        patch_size=(24, 192, 192),
        patch_num=20,
        intensity_threshold=0.12,
        area_ratio_threshold=0.032,
        raw_background_threshold=0.15,
        max_attempts=None,
        rng=None):
    """Generate accepted paired 3D patches with shared crop coordinates."""
    if rng is None:
        rng = random.Random()
    if max_attempts is None:
        max_attempts = max(patch_num * 1000, 1000)

    raw_patch = []
    gt_patch = []
    z_size, y_size, x_size = raw.shape
    patch_z, patch_y, patch_x = patch_size
    area_threshold = area_ratio_threshold * np.prod(patch_size)

    if raw.shape != gt.shape:
        raise ValueError(f'Raw and GT shapes must match, but got {raw.shape} and {gt.shape}.')
    if patch_z > z_size or patch_y > y_size or patch_x > x_size:
        raise ValueError(f'Patch size {patch_size} is larger than volume shape {raw.shape}.')

    attempts = 0
    while len(raw_patch) < patch_num and attempts < max_attempts:
        attempts += 1
        zt = rng.randint(0, z_size - patch_z)
        yt = rng.randint(0, y_size - patch_y)
        xt = rng.randint(0, x_size - patch_x)

        raw_crop = raw[zt:zt + patch_z, yt:yt + patch_y, xt:xt + patch_x]
        gt_crop = gt[zt:zt + patch_z, yt:yt + patch_y, xt:xt + patch_x]

        if np.max(raw_crop) < raw_background_threshold:
            continue

        if np.count_nonzero(gt_crop > intensity_threshold) <= area_threshold:
            continue

        raw_patch.append(raw_crop)
        gt_patch.append(gt_crop)

    if len(raw_patch) < patch_num:
        raise RuntimeError(
            f'Only accepted {len(raw_patch)}/{patch_num} patches after {attempts} attempts. '
            f'Consider lowering foreground/background thresholds or increasing max attempts.')

    return raw_patch, gt_patch


def safe_filename_token(value):
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', value)
    token = token.strip('._-')
    return token or 'item'


def ensure_output_dirs(output_root, split):
    low_dir = output_root / split / 'low'
    gt_dir = output_root / split / 'gt'
    low_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    return low_dir, gt_dir


def write_patch_pair(low_dir, gt_dir, stem, raw_patch, gt_patch, overwrite):
    raw_path = low_dir / f'{stem}.tif'
    gt_path = gt_dir / f'{stem}.tif'
    if not overwrite and (raw_path.exists() or gt_path.exists()):
        raise FileExistsError(
            f'Output file already exists: {raw_path if raw_path.exists() else gt_path}. '
            f'Use --overwrite to replace existing files.')
    if raw_patch.shape != gt_patch.shape:
        raise ValueError(f'Raw and GT patch shapes differ before writing: {raw_patch.shape} vs {gt_patch.shape}')

    imwrite(raw_path, raw_patch)
    imwrite(gt_path, gt_patch)


def generate_dataset(args):
    validate_percentiles(args.percentile_low, args.percentile_high)
    input_root = args.input_root
    output_root = args.output_root
    splits = normalize_splits(args.split)
    patch_size = tuple(args.patch_size)
    rng = random.Random(args.seed)

    if not input_root.is_dir():
        raise FileNotFoundError(f'Input root does not exist or is not a directory: {input_root}')

    requested_categories = parse_categories(args.categories)
    categories = resolve_categories(input_root, requested_categories)
    for split in splits:
        pairs_by_category = collect_all_pairs(input_root, categories, split)
        pair_quotas = build_pair_quotas(args.num_patches, pairs_by_category)
        low_dir, gt_dir = ensure_output_dirs(output_root, split)

        patch_index = 1
        for pair, quota in pair_quotas:
            raw, gt = load_and_prepare_pair(
                pair,
                patch_size=patch_size,
                discard_z=args.discard_z,
                discard_xy=args.discard_xy,
                percentile_low=args.percentile_low,
                percentile_high=args.percentile_high)

            per_pair_max_attempts = args.max_attempts if args.max_attempts is not None else max(quota * 1000, 1000)
            try:
                raw_patches, gt_patches = generate_3d_patch_v2(
                    raw,
                    gt,
                    patch_size=patch_size,
                    patch_num=quota,
                    intensity_threshold=args.intensity_threshold,
                    area_ratio_threshold=args.area_ratio_threshold,
                    raw_background_threshold=args.raw_background_threshold,
                    max_attempts=per_pair_max_attempts,
                    rng=rng)
            except RuntimeError as error:
                raise RuntimeError(f'Failed to generate patches for {split}/{pair.category}/{pair.key}: {error}') from error

            category_token = safe_filename_token(pair.category)
            key_token = safe_filename_token(pair.key)
            for raw_patch, gt_patch in zip(raw_patches, gt_patches):
                stem = f'{patch_index:06d}_{category_token}_{key_token}'
                write_patch_pair(low_dir, gt_dir, stem, raw_patch, gt_patch, args.overwrite)
                patch_index += 1

            print(f'Generated {quota} patches from {split}/{pair.category}/{pair.key}')

        print(f'Generated {patch_index - 1} paired patches into: {output_root / split}')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Generate paired 3D TIFF patches from raw/GT microscopy stacks.')
    parser.add_argument('--input-root', type=Path, required=True,
                        help='Root folder containing category folders such as Actin, ER, and Golji.')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Output root. The script writes <output-root>/<split>/low and <output-root>/<split>/gt.')
    parser.add_argument('--split', nargs='+', choices=('train', 'test', 'all', 'both'), required=True,
                        help='Dataset split to generate. Use "all", "both", or "train test" to generate train then test.')
    parser.add_argument('--categories', nargs='+', default=['all'],
                        help='One or more category names, comma-separated names, or "all".')
    parser.add_argument('--num-patches', type=positive_int, required=True,
                        help='Total number of accepted paired patches across selected categories.')
    parser.add_argument('--patch-size', type=positive_int, nargs=3, metavar=('Z', 'Y', 'X'),
                        default=(16, 192, 192), help='3D patch size in Z Y X order.')
    parser.add_argument('--discard-z', type=nonnegative_int, default=2,
                        help='Number of z slices discarded from each side before normalization and patching.')
    parser.add_argument('--discard-xy', type=nonnegative_int, default=10,
                        help='Number of y/x pixels discarded from each side before normalization and patching.')
    parser.add_argument('--percentile-low', type=float, default=0.01,
                        help='Lower percentile for whole-volume normalization.')
    parser.add_argument('--percentile-high', type=float, default=99.99,
                        help='Upper percentile for whole-volume normalization.')
    parser.add_argument('--intensity-threshold', type=float, default=0.11,
                        help='GT intensity threshold used by the foreground pixel ratio filter.')
    parser.add_argument('--area-ratio-threshold', type=float, default=0.008,
                        help='Minimum foreground voxel ratio required for accepting a patch.')
    parser.add_argument('--raw-background-threshold', type=float, default=0.15,
                        help='Reject patches whose Raw max intensity is below this threshold.')
    parser.add_argument('--max-attempts', type=positive_int, default=None,
                        help='Maximum random crop attempts per source stack. Default is max(quota * 1000, 1000).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducible crop coordinates.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing paired output files.')
    return parser


def main():
    parser = build_parser()
    if USE_IDE_CONFIG_WHEN_NO_CLI_ARGS and len(sys.argv) == 1:
        print('No command-line arguments detected. Using IDE_CONFIG in generate_data_patches.py.')
        args = args_from_ide_config(IDE_CONFIG)
    else:
        args = parser.parse_args()
    generate_dataset(args)


if __name__ == '__main__':
    main()
