"""Six labeled sources, explicit manifests, aligned image/depth transforms.

A JSONL record has dataset, split, image, depth, depth_scale. Optional mask is
an externally supplied benchmark-validity/crop mask; no GT file is guessed.
Depth may be .npy (H,W) or a single-channel image. Scale converts stored labels
to positive depth units. Paths are relative to the manifest, or absolute.
"""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import pil_to_tensor

TRAIN_SOURCES = ('BlendedMVS', 'DIML', 'HRWSI', 'IRS', 'MegaDepth', 'TartanAir')
EVAL_SPLITS = {'KITTI': 'eigen_test', 'NYUv2': 'official_test', 'ETH3D': 'all_available_gt'}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def normalize_rgb(image):
    shape = (1, 3, 1, 1) if image.ndim == 4 else (3, 1, 1)
    return (image - image.new_tensor(MEAN).view(shape)) / image.new_tensor(STD).view(shape)


def resize_rgb(image, short_side=518):
    h, w = image.shape[-2:]
    ratio = short_side / min(h, w)
    hw = (max(short_side, round(h * ratio)), max(short_side, round(w * ratio)))
    return F.interpolate(image.unsqueeze(0), size=hw, mode='bicubic',
                         align_corners=False, antialias=True)[0].clamp(0, 1)


def prepare_image(path, short_side=518):
    with Image.open(path) as im:
        image = pil_to_tensor(im.convert('RGB')).float() / 255
    return normalize_rgb(resize_rgb(image, short_side)), image.shape[-2:]


def depth_to_inverse(depth, valid, normalize):
    inverse = torch.zeros_like(depth)
    inverse[valid] = depth[valid].reciprocal()
    if normalize:
        values = inverse[valid]
        if values.numel() == 0:
            raise ValueError('No valid depth annotations')
        span = values.max() - values.min()
        if span <= 0:
            raise ValueError('Constant target has no relative-depth supervision')
        inverse[valid] = (values - values.min()) / span
    return inverse


class RelativeDepthDataset(Dataset):
    def __init__(self, manifests, training, short_side=518, jitter=0.1):
        self.training, self.short_side = training, short_side
        self.jitter = ColorJitter(jitter, jitter, jitter)
        self.records = []
        seen = set()
        for name in manifests:
            path = Path(name).resolve()
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ('dataset', 'split', 'image', 'depth', 'depth_scale'):
                    if key not in row:
                        raise ValueError(f'{path}:{line_no}: missing {key}')
                if training:
                    if row['dataset'] not in TRAIN_SOURCES or row['split'] != 'train':
                        raise ValueError('Training must contain only the six labeled training sources')
                elif EVAL_SPLITS.get(row['dataset']) != row['split']:
                    raise ValueError('Evaluation requires KITTI Eigen test, NYUv2 official test, or ETH3D all GT')
                if not math.isfinite(row['depth_scale']) or row['depth_scale'] <= 0:
                    raise ValueError('depth_scale must be a positive finite multiplier')
                for key in ('image', 'depth', 'mask'):
                    if key in row:
                        row[key] = str((path.parent / row[key]).resolve())
                        if not Path(row[key]).is_file():
                            raise FileNotFoundError(row[key])
                identity = row['image']
                if identity in seen:
                    raise ValueError(f'Duplicate image record: {identity}')
                seen.add(identity)
                self.records.append(row)
        if not self.records:
            raise ValueError('Empty dataset manifest')
        if training and {x['dataset'] for x in self.records} != set(TRAIN_SOURCES):
            raise ValueError('The paper training run requires all six labeled sources')

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _array(path):
        if Path(path).suffix.lower() == '.npy':
            return np.load(path, allow_pickle=False)
        with Image.open(path) as im:
            return np.array(im)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row['image']) as im:
            image = pil_to_tensor(im.convert('RGB')).float() / 255
        array = self._array(row['depth'])
        if array.ndim != 2 or tuple(array.shape) != tuple(image.shape[-2:]):
            raise ValueError('GT depth must be a 2D array aligned to the original RGB resolution')
        depth = torch.from_numpy(np.array(array, dtype=np.float32)).unsqueeze(0) * row['depth_scale']
        valid = torch.isfinite(depth) & (depth > 0)
        if 'mask' in row:
            mask = torch.from_numpy(np.array(self._array(row['mask']), copy=True)).bool().unsqueeze(0)
            if mask.shape != valid.shape:
                raise ValueError('Benchmark mask must match original depth shape')
            valid &= mask
        if 'min_depth' in row:
            valid &= depth > row['min_depth']
        if 'max_depth' in row:
            valid &= depth < row['max_depth']
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        image = resize_rgb(image, self.short_side)
        if self.training:
            hw = image.shape[-2:]
            depth = F.interpolate(depth[None], size=hw, mode='nearest')[0]
            valid = F.interpolate(valid[None].float(), size=hw, mode='nearest')[0].bool()
            # Normalize on the valid full resized image before crop, as label preprocessing.
            inverse = depth_to_inverse(depth, valid, normalize=True)
            h, w = hw
            top = torch.randint(h - self.short_side + 1, ()).item()
            left = torch.randint(w - self.short_side + 1, ()).item()
            sl = (slice(None), slice(top, top+self.short_side), slice(left, left+self.short_side))
            image, depth, valid, inverse = (x[sl] for x in (image, depth, valid, inverse))
            if torch.rand(()) < .5:
                image, depth, valid, inverse = (x.flip(-1) for x in (image, depth, valid, inverse))
            image = self.jitter(image)
            if not valid.any():
                raise ValueError(f'Crop has no valid depth: {row["image"]}')
        else:
            inverse = depth_to_inverse(depth, valid, normalize=False)
        return {'image': normalize_rgb(image), 'image_raw': image, 'depth': depth,
                'inverse_depth': inverse, 'valid_mask': valid, 'path': row['image'],
                'dataset': row['dataset'], 'split': row['split']}
